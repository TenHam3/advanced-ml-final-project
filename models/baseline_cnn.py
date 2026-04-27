from torch import nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
   def __init__(self, in_channels, num_classes, img_size=(28, 28)):

       """
       Building blocks of convolutional neural network.

       Parameters:
           * in_channels: Number of channels in the input image (for grayscale images, 1)
           * num_classes: Number of classes to predict. In our problem, 10.
           * img_size: (H, W) of input images. Used to compute the FC layer input size.
       """
       super(BaselineCNN, self).__init__()

       # 1st convolutional layer
       self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
       self.bn1 = nn.BatchNorm2d(8)
       # Max pooling layer
       self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
       # 2nd convolutional layer
       self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
       self.bn2 = nn.BatchNorm2d(16)

    #    self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
    #    self.bn3 = nn.BatchNorm2d(16)
       # Fully connected layer — size depends on image dimensions after 2x MaxPool (each halves H and W)
       fc_input_size = 16 * (img_size[0] // 4) * (img_size[1] // 4)
       self.fc1 = nn.Linear(fc_input_size, num_classes)

       # 1x1 projection to match channels on the skip path when in != out
       self.shortcut1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=1)
       
       # 1x1 projection to match channels on the skip path when in != out
       self.shortcut2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1)

       # self.shortcut3 = nn.Identity()

   def forward(self, x):
       """
       Define the forward pass of the neural network.

       Parameters:
           x: Input tensor.

       Returns:
           torch.Tensor
               The output tensor after passing through the network.
       """
       x = F.relu(self.bn1(self.conv1(x) + self.shortcut1(x)))  # Apply first convolution and ReLU activation
       x = self.maxpool(x)           # Apply max pooling
       x = F.relu(self.bn2(self.conv2(x) + self.shortcut2(x)))  # Apply second convolution and ReLU activation
       x = self.maxpool(x)           # Apply max pooling
    #    x = F.relu(self.bn3(self.conv3(x) + self.shortcut3(x)))  # Apply second convolution and ReLU activation
    #    x = self.maxpool(x)           # Apply max pooling
       x = x.reshape(x.shape[0], -1)  # Flatten the tensor
       x = self.fc1(x)            # Apply fully connected layer
       return x