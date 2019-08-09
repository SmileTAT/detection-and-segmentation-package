from torch import nn


class FCN8s(nn.Module):
  """Builds fcn8s model based on vgg16 backbone.
      without batch normalization. """

  def __init__(self, num_classes=21):
    super(FCN8s, self).__init__()
    self.num_classes = num_classes

    # conv1
    self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)  # padding=100, avoid the situation that feature maps are too samll after pooling
    self.relu1_1 = nn.ReLU(inplace=True)
    self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
    self.relu1_2 = nn.ReLU(inplace=True)
    self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)  # 1/2

    # conv2
    self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
    self.relu2_1 = nn.ReLU(inplace=True)
    self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
    self.relu2_2 = nn.ReLU(inplace=True)
    self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)  # 1/4

    # conv3
    self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
    self.relu3_1 = nn.ReLU(inplace=True)
    self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
    self.relu3_2 = nn.ReLU(inplace=True)
    self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
    self.relu3_3 = nn.ReLU(inplace=True)
    self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)  # 1/8

    # conv4
    self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
    self.relu4_1 = nn.ReLU(inplace=True)
    self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
    self.relu4_2 = nn.ReLU(inplace=True)
    self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
    self.relu4_3 = nn.ReLU(inplace=True)
    self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)  # 1/16

    # conv5
    self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
    self.relu5_1 = nn.ReLU(inplace=True)
    self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
    self.relu5_2 = nn.ReLU(inplace=True)
    self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
    self.relu5_3 = nn.ReLU(inplace=True)
    self.pool5 = nn.MaxPool2d(2, 2, ceil_mode=True)  # 1/32

    # fc6
    self.fc6 = nn.Conv2d(512, 4096, 7)
    self.relu6 = nn.ReLU(inplace=True)

    # fc7
    self.fc7 = nn.Conv2d(4096, 4096, 1)
    self.relu7 = nn.ReLU(inplace=True)

    # score
    self.score_pool3 = nn.Conv2d(256, self.num_classes, 1)
    self.score_pool4 = nn.Conv2d(512, self.num_classes, 1)
    self.score = nn.Conv2d(4096, self.num_classes, 1)

    self.upscore2 = nn.ConvTranspose2d(self.num_classes, self.num_classes, 4, stride=2, bias=False)
    self.upscore_pool4 = nn.ConvTranspose2d(self.num_classes, self.num_classes, 4, stride=2, bias=False)
    self.upscore8 = nn.ConvTranspose2d(self.num_classes, self.num_classes, 32, stride=8, bias=False)

  def forward(self, x):
    h = self.relu1_1(self.conv1_1(x))
    h = self.relu1_1(self.conv1_2(h))
    h = self.pool1(h)

    h = self.relu2_1(self.conv2_1(h))
    h = self.relu2_2(self.conv2_2(h))
    h = self.pool2(h)

    h = self.relu3_1(self.conv3_1(h))
    h = self.relu3_2(self.conv3_2(h))
    h = self.relu3_3(self.conv3_3(h))
    h = self.pool3(h)
    pool3 = h

    h = self.relu4_1(self.conv4_1(h))
    h = self.relu4_2(self.conv4_2(h))
    h = self.relu4_3(self.conv4_3(h))
    h = self.pool4(h)
    pool4 = h

    h = self.relu5_1(self.conv5_1(h))
    h = self.relu5_2(self.conv5_2(h))
    h = self.relu5_3(self.conv5_3(h))
    h = self.pool5(h)

    h = self.relu6(self.fc6(h))

    h = self.relu7(self.fc7(h))

    h = self.score(h)

    upscore2 = self.upscore2(h)
    pool4_score = self.score_pool4(pool4)[:, :, 5:5+upscore2.size()[2], 5:5+upscore2.size()[3]]

    h = upscore2 + pool4_score
    upscore_pool4 = self.upscore_pool4(h)

    pool3_score = self.score_pool3(pool3)[:, :, 9:9+upscore_pool4.size()[2], 9:9+upscore_pool4.size()[3]]

    h = upscore_pool4 + pool3_score
    h = self.upscore8(h)

    h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

    return h
