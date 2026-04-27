import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchmetrics.classification import Accuracy

from models.baseline_cnn import BaselineCNN
from models.test_cnn import TestCNN

# ── Dataset selection ─────────────────────────────────────────────────────────
DATASET = "CIFAR10"  # Options: "MNIST", "CIFAR10", "STL10"

DATASET_CONFIGS = {
    "MNIST": {
        "class":        datasets.MNIST,
        "train_kwargs": {"train": True},
        "test_kwargs":  {"train": False},
        "in_channels":  1,
        "num_classes":  10,
        "img_size":     (28, 28),
        "transform":    transforms.ToTensor(),
        "train_subset": 5000,
        "test_subset":  1000,
    },
    "CIFAR10": {
        "class":        datasets.CIFAR10,
        "train_kwargs": {"train": True},
        "test_kwargs":  {"train": False},
        "in_channels":  3,
        "num_classes":  10,
        "img_size":     (32, 32),
        "transform":    transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                 (0.2023, 0.1994, 0.2010)),
                        ]),
        "train_subset": 5000,
        "test_subset":  1000,
    },
    "STL10": {
        "class":        datasets.STL10,
        "train_kwargs": {"split": "train"},
        "test_kwargs":  {"split": "test"},
        "in_channels":  3,
        "num_classes":  10,
        "img_size":     (96, 96),
        "transform":    transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4467, 0.4398, 0.4066),
                                                 (0.2603, 0.2566, 0.2713)),
                        ]),
        "train_subset": 5000,
        "test_subset":  1000,
    },
}

# ── Hyperparameters ───────────────────────────────────────────────────────────
NUM_RUNS        = 5
PRETRAIN_EPOCHS = 5   # epochs to train baseline (and TestCNN-scratch for fair budget)
PHASE_A_EPOCHS  = 5    # fine-tune epochs: conv frozen, only alpha + FC trained
PHASE_B_EPOCHS  = 5   # fine-tune epochs: all params, differential lr

PRETRAIN_LR = 0.0005
ALPHA_LR    = 0.001    # alpha adapts faster than the conv kernel
CONV_LR     = 0.00005  # low lr keeps pretrained kernel from drifting too fast
FC_LR       = 0.0005
# ─────────────────────────────────────────────────────────────────────────────


def get_loaders(cfg):
    train_ds = cfg["class"](root="dataset/", download=True,
                            transform=cfg["transform"], **cfg["train_kwargs"])
    test_ds  = cfg["class"](root="dataset/", download=True,
                            transform=cfg["transform"], **cfg["test_kwargs"])
    train_loader = DataLoader(Subset(train_ds, range(cfg["train_subset"])),
                              batch_size=10, shuffle=True)
    test_loader  = DataLoader(Subset(test_ds,  range(cfg["test_subset"])),
                              batch_size=10, shuffle=False)
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for data, targets in tqdm(loader, leave=False):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        criterion(model(data), targets).backward()
        optimizer.step()


def evaluate(model, loader, num_classes, device):
    acc = Accuracy(task="multiclass", num_classes=num_classes)
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(dim=1).cpu()
            acc(preds, labels)
    return acc.compute().item()


def transfer_weights(src: BaselineCNN, dst: TestCNN):
    """Copy conv1/conv2 weights from a trained BaselineCNN into TestCNN's shared kernels."""
    with torch.no_grad():
        dst.layer1.conv.weight.copy_(src.conv1.weight)
        dst.layer1.conv.bias.copy_(src.conv1.bias)
        dst.layer2.conv.weight.copy_(src.conv2.weight)
        dst.layer2.conv.bias.copy_(src.conv2.bias)
        # Neutral dilation start: uniform 1/3 weight on each dilation
        nn.init.zeros_(dst.layer1.alpha)
        nn.init.zeros_(dst.layer2.alpha)
        # FC sees different feature distribution after multi-dilation blend, so reinit
        nn.init.kaiming_normal_(dst.fc1.weight)
        nn.init.zeros_(dst.fc1.bias)


def print_dilation_preferences(model: TestCNN, dilations=(1, 2, 3)):
    """Print which dilation rate each channel preferred after fine-tuning."""
    for name, layer in [("Layer 1", model.layer1), ("Layer 2", model.layer2)]:
        weights = F.softmax(layer.alpha.detach().cpu(), dim=1)  # (C, num_dilations)
        preferred = weights.argmax(dim=1)                        # (C,)
        counts = {d: (preferred == i).sum().item() for i, d in enumerate(dilations)}
        avg_w = weights.mean(dim=0)
        print(f"  {name} dilation preference (out of {weights.shape[0]} channels):")
        for i, d in enumerate(dilations):
            print(f"    dilation={d}: {counts[d]:2d} channels prefer it  "
                  f"(avg weight {avg_w[i]:.3f})")


# ── Main experiment ───────────────────────────────────────────────────────────
cfg    = DATASET_CONFIGS[DATASET]
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device} | Dataset: {DATASET}\n")

# Results are collected across runs
# Comparison is:
#   baseline       — trained for PRETRAIN_EPOCHS (the model whose weights get transferred)
#   scratch        — TestCNN trained from random init for PRETRAIN_EPOCHS + PHASE_A + PHASE_B
#                    (same total epoch budget as pretrain + finetune combined)
#   pretrain+ft    — baseline weights transferred, then finetuned for PHASE_A + PHASE_B

baseline_accs = []
scratch_accs  = []
finetune_accs = []

total_finetune_epochs = PRETRAIN_EPOCHS + PHASE_A_EPOCHS + PHASE_B_EPOCHS

for run in range(NUM_RUNS):
    print(f"{'='*55}")
    print(f"Run {run+1}/{NUM_RUNS}")
    print(f"{'='*55}")

    train_loader, test_loader = get_loaders(cfg)
    criterion = nn.CrossEntropyLoss()

    # ── Stage 1: Pretrain baseline ────────────────────────────────────────────
    print(f"\n[Stage 1] Pretraining BaselineCNN for {PRETRAIN_EPOCHS} epochs...")
    baseline = BaselineCNN(cfg["in_channels"], cfg["num_classes"], cfg["img_size"]).to(device)
    opt = optim.Adam(baseline.parameters(), lr=PRETRAIN_LR)
    for epoch in range(PRETRAIN_EPOCHS):
        print(f"  Epoch [{epoch+1}/{PRETRAIN_EPOCHS}]", end="\r")
        train_one_epoch(baseline, train_loader, criterion, opt, device)
    baseline_acc = evaluate(baseline, test_loader, cfg["num_classes"], device)
    baseline_accs.append(baseline_acc)
    print(f"  Baseline accuracy: {baseline_acc:.4f}                   ")

    # ── Stage 2: TestCNN from scratch (matched epoch budget) ──────────────────
    print(f"\n[Stage 2] Training BaselineCNN from scratch for {total_finetune_epochs} epochs...")
    scratch_model = BaselineCNN(cfg["in_channels"], cfg["num_classes"], cfg["img_size"]).to(device)
    opt = optim.Adam(scratch_model.parameters(), lr=PRETRAIN_LR)
    for epoch in range(total_finetune_epochs):
        print(f"  Epoch [{epoch+1}/{total_finetune_epochs}]", end="\r")
        train_one_epoch(scratch_model, train_loader, criterion, opt, device)
    scratch_acc = evaluate(scratch_model, test_loader, cfg["num_classes"], device)
    scratch_accs.append(scratch_acc)
    print(f"  Baseline (scratch) accuracy: {scratch_acc:.4f}           ")

    # ── Stage 3: Transfer weights + two-phase fine-tuning ────────────────────
    print(f"\n[Stage 3] Fine-tuning TestCNN from pretrained baseline weights...")
    ft_model = TestCNN(cfg["in_channels"], cfg["num_classes"], cfg["img_size"]).to(device)
    transfer_weights(baseline, ft_model)

    # Phase A: conv frozen — only alpha and FC learn
    print(f"  Phase A: alpha + FC only, conv frozen ({PHASE_A_EPOCHS} epochs)...")
    for p in ft_model.layer1.conv.parameters():
        p.requires_grad = False
    for p in ft_model.layer2.conv.parameters():
        p.requires_grad = False
    opt = optim.Adam(filter(lambda p: p.requires_grad, ft_model.parameters()),
                     lr=ALPHA_LR)
    for epoch in range(PHASE_A_EPOCHS):
        print(f"    Epoch [{epoch+1}/{PHASE_A_EPOCHS}]", end="\r")
        train_one_epoch(ft_model, train_loader, criterion, opt, device)

    # Phase B: unfreeze conv, differential learning rates
    print(f"  Phase B: all params, differential lr ({PHASE_B_EPOCHS} epochs)...")
    for p in ft_model.layer1.conv.parameters():
        p.requires_grad = True
    for p in ft_model.layer2.conv.parameters():
        p.requires_grad = True
    opt = optim.Adam([
        {"params": ft_model.layer1.conv.parameters(), "lr": CONV_LR},
        {"params": ft_model.layer1.alpha,              "lr": ALPHA_LR},
        {"params": ft_model.layer2.conv.parameters(), "lr": CONV_LR},
        {"params": ft_model.layer2.alpha,              "lr": ALPHA_LR},
        {"params": ft_model.fc1.parameters(),          "lr": FC_LR},
    ])
    for epoch in range(PHASE_B_EPOCHS):
        print(f"    Epoch [{epoch+1}/{PHASE_B_EPOCHS}]", end="\r")
        train_one_epoch(ft_model, train_loader, criterion, opt, device)

    finetune_acc = evaluate(ft_model, test_loader, cfg["num_classes"], device)
    finetune_accs.append(finetune_acc)
    print(f"  TestCNN (pretrained+finetuned) accuracy: {finetune_acc:.4f}")

    print("\n  Learned dilation preferences after fine-tuning:")
    print_dilation_preferences(ft_model)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("RESULTS SUMMARY")
print(f"  Dataset: {DATASET}")
print(f"  Baseline epochs:      {PRETRAIN_EPOCHS}")
print(f"  Scratch epochs:       {total_finetune_epochs}  (matched budget)")
print(f"  Pretrain+ft epochs:   {PRETRAIN_EPOCHS} pretrain + "
      f"{PHASE_A_EPOCHS}A + {PHASE_B_EPOCHS}B finetune")
print(f"{'='*55}")

for label, accs in [
    ("Baseline CNN        ", baseline_accs),
    ("TestCNN (scratch)   ", scratch_accs),
    ("TestCNN (pretrain+ft)", finetune_accs),
]:
    avg   = sum(accs) / NUM_RUNS
    best  = max(accs)
    worst = min(accs)
    print(f"\n{label}  avg={avg:.4f}  best={best:.4f}  worst={worst:.4f}")
    print(f"  per-run: {[f'{a:.4f}' for a in accs]}")
