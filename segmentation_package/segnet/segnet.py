from torch import nn
import torch.nn.functional as F


class SegNet(nn.Module):
  """Builds SegNet based on VGG16 backbone. """

  def __init__(self, num_classes=21):
    super(SegNet, self).__init__()
    self.num_classes = num_classes

    # conv1
    self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
    self.relu1_1 = nn.ReLU(inplace=True)
    self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
    self.relu1_2 = nn.ReLU(inplace=True)
    self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)

    # conv2
    self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
    self.relu2_1 = nn.ReLU(inplace=True)
    self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
    self.relu2_2 = nn.ReLU(inplace=True)
    self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)

    # conv3
    self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
    self.relu3_1 = nn.ReLU(inplace=True)
    self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
    self.relu3_2 = nn.ReLU(inplace=True)
    self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
    self.relu3_3 = nn.ReLU(inplace=True)
    self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)

    # conv4
    self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
    self.relu4_1 = nn.ReLU(inplace=True)
    self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
    self.relu4_2 = nn.ReLU(inplace=True)
    self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
    self.relu4_3 = nn.ReLU(inplace=True)
    self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)

    # conv5
    self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
    self.relu5_1 = nn.ReLU(inplace=True)
    self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
    self.relu5_2 = nn.ReLU(inplace=True)
    self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
    self.relu5_3 = nn.ReLU(inplace=True)
    self.pool5 = nn.MaxPool2d(2, 2, return_indices=True)



  def forward(self, x):
    h = self.relu1_1(self.conv1_1(x))
    h = self.relu1_2(self.conv1_2(h))
    pool1, index1 = self.pool1(h)

    h = self.relu2_1(self.conv2_1(pool1))
    h = self.relu2_2(self.conv2_2(h))
    pool2, index2 = self.pool2(h)

    h = self.relu3_1(self.conv3_1(pool2))
    h = self.relu3_2(self.conv3_2(h))
    h = self.relu3_2(self.conv3_2(h))
    pool3, index3 = self.pool3(h)

    h = self.relu4_1(self.conv4_1(pool3))
    h = self.relu4_2(self.conv4_2(h))
    h = self.relu4_2(self.conv4_2(h))
    pool4, index4 = self.pool4(h)

    h = self.relu5_1(self.conv5_1(pool3))
    h = self.relu5_2(self.conv5_2(h))
    h = self.relu5_2(self.conv5_2(h))
    pool5, index5 = self.pool4(h)

    uppool5 = F.max_unpool2d(pool5, index5, kernel_size=2, stride=2)
