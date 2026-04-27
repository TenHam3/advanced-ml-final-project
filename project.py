import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import torchvision
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torchmetrics
from torchmetrics.classification import Accuracy, Precision, Recall
from models.baseline_cnn import BaselineCNN
from models.baseline_cifar import BaselineCIFAR
from models.baseline_stl import BaselineSTL
from models.test_cnn import TestCNN
from models.independent_cnn import IndependentCNN
from models.dilated_resnet_test import DilatedResNetTest
from models.dilated_resnet_independent import DilatedResNetIndependent

import time

# ─── Dataset selection ──────────────────────────────────────────────────────
DATASET = "CIFAR10"  # Options: "MNIST", "CIFAR10", "STL10"

DATASET_CONFIGS = {
    "MNIST": {
        "class":           datasets.MNIST,
        "train_kwargs":    {"train": True},
        "test_kwargs":     {"train": False},
        "in_channels":     1,
        "num_classes":     10,
        "img_size":        (28, 28),
        "train_transform": transforms.ToTensor(),
        "test_transform":  transforms.ToTensor(),
        "train_subset":    5000,
        "test_subset":     1000,
    },
    "CIFAR10": {
        "class":           datasets.CIFAR10,
        "train_kwargs":    {"train": True},
        "test_kwargs":     {"train": False},
        "in_channels":     3,
        "num_classes":     10,
        "img_size":        (32, 32),
        "train_transform": transforms.Compose([
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomCrop(32, padding=4),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                    (0.2023, 0.1994, 0.2010)),
                           ]),
        "test_transform":  transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                    (0.2023, 0.1994, 0.2010)),
                           ]),
        "train_subset":    5000,
        "test_subset":     1000,
    },
    "STL10": {
        "class":           datasets.STL10,
        "train_kwargs":    {"split": "train"},
        "test_kwargs":     {"split": "test"},
        "in_channels":     3,
        "num_classes":     10,
        "img_size":        (96, 96),
        "train_transform": transforms.Compose([
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomCrop(96, padding=12),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4467, 0.4398, 0.4066),
                                                    (0.2603, 0.2566, 0.2713)),
                           ]),
        "test_transform":  transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4467, 0.4398, 0.4066),
                                                    (0.2603, 0.2566, 0.2713)),
                           ]),
        "train_subset":    5000,
        "test_subset":     1000,
    },
}
# ────────────────────────────────────────────────────────────────────────────

cfg = DATASET_CONFIGS[DATASET]

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def get_loaders(cfg):
    train_dataset = cfg["class"](root="dataset/", download=True, transform=cfg["train_transform"], **cfg["train_kwargs"])
    train_loader = DataLoader(dataset=Subset(train_dataset, list(range(cfg["train_subset"]))), batch_size=batch_size, shuffle=True)
    test_dataset = cfg["class"](root="dataset/", download=True, transform=cfg["test_transform"], **cfg["test_kwargs"])
    test_loader = DataLoader(dataset=Subset(test_dataset, list(range(cfg["test_subset"]))), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

batch_size = 50 if DATASET != "MNIST" else 10
num_runs = 5
num_epochs = 50 if DATASET != "MNIST" else 10

device = "cuda" if torch.cuda.is_available() else "cpu"

baseline_accuracies, baseline_precision, baseline_recall, baseline_times = [], [], [], []
dilated_accuracies, dilated_precision, dilated_recall, dilated_times = [], [], [], []
ind_accuracies, ind_precision, ind_recall, ind_times = [], [], [], []

for i in range(num_runs):
    print(f"Run {i+1}/{num_runs}:")

    train_loader, test_loader = get_loaders(cfg)
    criterion = nn.CrossEntropyLoss()

    # ── Baseline ──────────────────────────────────────────────────────────────
    if DATASET == "MNIST":
        baseline_model = BaselineCNN(
            in_channels=cfg["in_channels"],
            num_classes=cfg["num_classes"],
            img_size=cfg["img_size"],
        ).to(device)
    elif DATASET == "CIFAR10":
        baseline_model = BaselineCIFAR(
            in_channels=cfg["in_channels"],
            num_classes=cfg["num_classes"],
            img_size=cfg["img_size"],
        ).to(device)
    else:
        baseline_model = BaselineSTL(
            in_channels=cfg["in_channels"],
            num_classes=cfg["num_classes"],
            img_size=cfg["img_size"],
        ).to(device)

    if DATASET == "MNIST":
        optimizer = optim.Adam(baseline_model.parameters(), lr=0.0005)
        scheduler = None
    else:
        optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    start = time.perf_counter()
    for epoch in range(num_epochs):
        baseline_model.train()
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            targets = targets.to(device)
            scores = baseline_model(data)
            loss = criterion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
    end = time.perf_counter()
    baseline_times.append(end - start)

    acc = Accuracy(task="multiclass", num_classes=cfg["num_classes"]).to(device)
    precision = Precision(task="multiclass", average='macro', num_classes=cfg["num_classes"]).to(device)
    recall = Recall(task="multiclass", average='macro', num_classes=cfg["num_classes"]).to(device)

    baseline_model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = baseline_model(images)
            _, preds = torch.max(outputs, 1)
            acc(preds, labels)
            precision(preds, labels)
            recall(preds, labels)

    test_accuracy = acc.compute()
    test_precision = precision.compute()
    test_recall = recall.compute()
    print(f"Baseline test accuracy: {test_accuracy}")
    print(f"Baseline test precision: {test_precision}")
    print(f"Baseline test recall: {test_recall}")
    print(f"Baseline train time: {end - start:.3f}")
    baseline_accuracies.append(test_accuracy)
    baseline_precision.append(test_precision)
    baseline_recall.append(test_recall)

    # ── TestCNN / DilatedResNetTest ───────────────────────────────────────────
    if DATASET == "MNIST":
        dilated_model = TestCNN(
            in_channels=cfg["in_channels"],
            num_classes=cfg["num_classes"],
            img_size=cfg["img_size"],
        ).to(device)
    else:
        dilated_model = DilatedResNetTest(
            in_channels=cfg["in_channels"],
            num_classes=cfg["num_classes"],
            img_size=cfg["img_size"],
        ).to(device)

    if DATASET == "MNIST":
        optimizer = optim.Adam(dilated_model.parameters(), lr=0.0005)
        scheduler = None
    else:
        optimizer = optim.Adam(dilated_model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    start = time.perf_counter()
    for epoch in range(num_epochs):
        dilated_model.train()
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            targets = targets.to(device)
            scores = dilated_model(data)
            loss = criterion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
    end = time.perf_counter()
    dilated_times.append(end - start)

    acc = Accuracy(task="multiclass", num_classes=cfg["num_classes"]).to(device)
    precision = Precision(task="multiclass", average='macro', num_classes=cfg["num_classes"]).to(device)
    recall = Recall(task="multiclass", average='macro', num_classes=cfg["num_classes"]).to(device)

    dilated_model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = dilated_model(images)
            _, preds = torch.max(outputs, 1)
            acc(preds, labels)
            precision(preds, labels)
            recall(preds, labels)

    test_accuracy = acc.compute()
    test_precision = precision.compute()
    test_recall = recall.compute()
    print(f"TestCNN test accuracy: {test_accuracy}")
    print(f"TestCNN test precision: {test_precision}")
    print(f"TestCNN test recall: {test_recall}")
    print(f"TestCNN train time: {end - start:.3f}")
    dilated_accuracies.append(test_accuracy)
    dilated_precision.append(test_precision)
    dilated_recall.append(test_recall)

    # ── IndependentCNN / DilatedResNetIndependent ─────────────────────────────
    if DATASET == "MNIST":
        ind_model = IndependentCNN(
            in_channels=cfg["in_channels"],
            num_classes=cfg["num_classes"],
            img_size=cfg["img_size"],
        ).to(device)
    else:
        ind_model = DilatedResNetIndependent(
            in_channels=cfg["in_channels"],
            num_classes=cfg["num_classes"],
            img_size=cfg["img_size"],
        ).to(device)

    if DATASET == "MNIST":
        optimizer = optim.Adam(ind_model.parameters(), lr=0.0005)
        scheduler = None
    else:
        optimizer = optim.Adam(ind_model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    start = time.perf_counter()
    for epoch in range(num_epochs):
        ind_model.train()
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            targets = targets.to(device)
            scores = ind_model(data)
            loss = criterion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
    end = time.perf_counter()
    ind_times.append(end - start)

    acc = Accuracy(task="multiclass", num_classes=cfg["num_classes"]).to(device)
    precision = Precision(task="multiclass", average='macro', num_classes=cfg["num_classes"]).to(device)
    recall = Recall(task="multiclass", average='macro', num_classes=cfg["num_classes"]).to(device)

    ind_model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = ind_model(images)
            _, preds = torch.max(outputs, 1)
            acc(preds, labels)
            precision(preds, labels)
            recall(preds, labels)

    test_accuracy = acc.compute()
    test_precision = precision.compute()
    test_recall = recall.compute()
    print(f"IndependentCNN test accuracy: {test_accuracy}")
    print(f"IndependentCNN test precision: {test_precision}")
    print(f"IndependentCNN test recall: {test_recall}")
    print(f"IndependentCNN train time: {end - start:.3f}")
    ind_accuracies.append(test_accuracy)
    ind_precision.append(test_precision)
    ind_recall.append(test_recall)

print(f"\nAverage baseline accuracy: {sum(baseline_accuracies) / num_runs}\nAverage baseline training time: {sum(baseline_times) / num_runs}")
print(f"Average baseline precision: {sum(baseline_precision) / num_runs}")
print(f"Average baseline recall: {sum(baseline_recall) / num_runs}")
print(f"Best baseline accuracy: {max(baseline_accuracies)}")
print(f"Worst baseline accuracy: {min(baseline_accuracies)}")

print(f"\nAverage dilated accuracy: {sum(dilated_accuracies) / num_runs}\nAverage dilated training time: {sum(dilated_times) / num_runs}")
print(f"Average dilated precision: {sum(dilated_precision) / num_runs}")
print(f"Average dilated recall: {sum(dilated_recall) / num_runs}")
print(f"Best dilated accuracy: {max(dilated_accuracies)}")
print(f"Worst dilated accuracy: {min(dilated_accuracies)}")

print(f"\nAverage ind accuracy: {sum(ind_accuracies) / num_runs}\nAverage ind training time: {sum(ind_times) / num_runs}")
print(f"Average ind precision: {sum(ind_precision) / num_runs}")
print(f"Average ind recall: {sum(ind_recall) / num_runs}")
print(f"Best ind accuracy: {max(ind_accuracies)}")
print(f"Worst ind accuracy: {min(ind_accuracies)}")