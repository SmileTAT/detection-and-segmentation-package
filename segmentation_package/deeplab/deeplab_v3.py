import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet34


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_channels, channels, stride=1, dilation=1):
    super(BasicBlock, self).__init__()

    out_channels = self.expansion * channels

    self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(channels)

    self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
    self.bn2 = nn.BatchNorm2d(channels)

    if (stride != 1) or (in_channels != out_channels):
      conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
      bn = nn.BatchNorm2d(out_channels)
      self.downsample = nn.Sequential(conv, bn)
    else:
      self.downsample = nn.Sequential()

  def forward(self, x):
    # (x has shape: (batch_size, in_channels, h, w))

    out = F.relu(self.bn1(self.conv1(
      x)))  # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)
    out = self.bn2(self.conv2(
      out))  # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

    out = out + self.downsample(
      x)  # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

    out = F.relu(
      out)  # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

    return out


class ASPP(nn.Module):
  """Build ASPP module for deeplab v3. """

  def __init__(self, in_channels, out_channels):
    super(ASPP, self).__init__()
    self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, 1)
    self.bn_conv_1x1_1 = nn.BatchNorm2d(out_channels)

    self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
    self.bn_conv_3x3_1 = nn.BatchNorm2d(out_channels)

    self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
    self.bn_conv_3x3_2 = nn.BatchNorm2d(out_channels)

    self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18)
    self.bn_conv_3x3_3 = nn.BatchNorm2d(out_channels)

    self.avg_pool = nn.AdaptiveAvgPool2d(1)  # to overcome the problem that when the dilate
    # rate is close to feature map size, the conv 3x3 degenerates to a simple conv_1x1, only
    # the center filter weight is effective.
    self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, 1)
    self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)

    self.conv_1x1_3 = nn.Conv2d(5 * out_channels, out_channels, 1)  # conv_1x1 for concat features
    self.bn_conv_1x1_3 = nn.BatchNorm2d(out_channels)

  def forward(self, x):
    h_conv_1x1_1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x)), inplace=True)
    h_conv_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(x)), inplace=True)
    h_conv_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(x)), inplace=True)
    h_conv_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(x)), inplace=True)
    h_avg_pool = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(self.avg_pool(x))), inplace=True)
    h_avg_pool = F.interpolate(h_avg_pool, x.size()[2:], mode='bilinear', align_corners=True)

    out = torch.cat([h_conv_1x1_1, h_conv_3x3_1, h_conv_3x3_2, h_conv_3x3_3, h_avg_pool], dim=1)
    out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)), inplace=True)
    return out


class DeepLabV3(nn.Module):
  """Build deeplab v3 based on resnet34 for segmentation task. """

  def __init__(self, num_classes=21):
    super(DeepLabV3, self).__init__()
    self.num_classes = num_classes

    self.backbone = nn.Sequential(*list(resnet34().children())[:-3])
    self.layer5 = self.make_layer(BasicBlock, in_channels=256, channels=512, num_blocks=3, stride=1, dilation=2)

    self.aspp = ASPP(512, 256)

    self.final = nn.Conv2d(256, self.num_classes, 1)

  def forward(self, x):
    h = self.backbone(x)
    h = self.layer5(h)

    h = self.aspp(h)

    h = self.final(h)  # 1/16

    out = F.interpolate(h, x.size()[2:], mode='bilinear', align_corners=True)
    return out

  @staticmethod
  def make_layer(block, in_channels, channels, num_blocks, stride=1, dilation=1):
    strides = [stride] + [1] * (num_blocks - 1)  # (stride == 2, num_blocks == 4 --> strides == [2, 1, 1, 1])

    blocks = []
    for stride in strides:
      blocks.append(block(in_channels=in_channels, channels=channels, stride=stride, dilation=dilation))
      in_channels = block.expansion * channels

    layer = nn.Sequential(*blocks)

    return layer
