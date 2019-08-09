from torch import nn


class SegNet(nn.Module):
  """Builds SegNet based on VGG16 backbone.
      without batch normalization. """

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

    # conv5d
    self.pool5d = nn.MaxUnpool2d(2, 2)
    self.conv5_3d = nn.Conv2d(512, 512, 3, padding=1)
    self.relu5_3d = nn.ReLU(inplace=True)
    self.conv5_2d = nn.Conv2d(512, 512, 3, padding=1)
    self.relu5_2d = nn.ReLU(inplace=True)
    self.conv5_1d = nn.Conv2d(512, 512, 3, padding=1)
    self.relu5_1d = nn.ReLU(inplace=True)

    # conv4d
    self.pool4d = nn.MaxUnpool2d(2, 2)
    self.conv4_3d = nn.Conv2d(512, 512, 3, padding=1)
    self.relu4_3d = nn.ReLU(inplace=True)
    self.conv4_2d = nn.Conv2d(512, 512, 3, padding=1)
    self.relu4_2d = nn.ReLU(inplace=True)
    self.conv4_1d = nn.Conv2d(512, 256, 3, padding=1)
    self.relu4_1d = nn.ReLU(inplace=True)

    # conv3d
    self.pool3d = nn.MaxUnpool2d(2, 2)
    self.conv3_3d = nn.Conv2d(256, 256, 3, padding=1)
    self.relu3_3d = nn.ReLU(inplace=True)
    self.conv3_2d = nn.Conv2d(256, 256, 3, padding=1)
    self.relu3_2d = nn.ReLU(inplace=True)
    self.conv3_1d = nn.Conv2d(256, 128, 3, padding=1)
    self.relu3_1d = nn.ReLU(inplace=True)

    # conv2d
    self.pool2d = nn.MaxUnpool2d(2, 2)
    self.conv2_2d = nn.Conv2d(128, 128, 3, padding=1)
    self.relu2_2d = nn.ReLU(inplace=True)
    self.conv2_1d = nn.Conv2d(128, 64, 3, padding=1)
    self.relu2_1d = nn.ReLU(inplace=True)

    # conv1d
    self.pool1d = nn.MaxUnpool2d(2, 2)
    self.conv1_2d = nn.Conv2d(64, 64, 3, padding=1)
    self.relu1_2d = nn.ReLU(inplace=True)
    self.conv1_1d = nn.Conv2d(64, self.num_classes, 3, padding=1)
    # self.relu1_1d = nn.ReLU(inplace=True)

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

    h = self.relu5_1(self.conv5_1(pool4))
    h = self.relu5_2(self.conv5_2(h))
    h = self.relu5_2(self.conv5_2(h))
    pool5, index5 = self.pool4(h)

    h = self.pool5d(pool5, index5)
    h = self.relu5_3d(self.conv5_3d(h))
    h = self.relu5_2d(self.conv5_2d(h))
    h = self.relu5_1d(self.conv5_1d(h))

    h = self.pool4d(h, index4)
    h = self.relu4_3d(self.conv4_3d(h))
    h = self.relu4_2d(self.conv4_2d(h))
    h = self.relu4_1d(self.conv4_1d(h))

    h = self.pool3d(h, index3)
    h = self.relu3_3d(self.conv3_3d(h))
    h = self.relu3_2d(self.conv3_2d(h))
    h = self.relu3_1d(self.conv3_1d(h))

    h = self.pool2d(h, index2)
    h = self.relu2_2d(self.conv2_2d(h))
    h = self.relu2_1d(self.conv2_1d(h))

    h = self.pool2d(h, index1)
    h = self.relu1_2d(self.conv1_2d(h))
    h = self.conv1_1d(h)
    return h



