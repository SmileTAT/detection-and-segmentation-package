from .base import BaseDetectionModel

from torch import nn


class RetinaNetBackbone(nn.Module):
  def __init__(self,
               name='RetinaNetBackbone',
               layers=50,
               ):
    super(RetinaNetBackbone, self).__init__()
    self.inplanes = 64
    self.conv1 = nn.Conv2d(in_channels=3,
                           out_channels=64,
                           kernel_size=7,
                           stride=2,
                           padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(num_features=64)
    self.relu1 = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3,
                                stride=2,
                                padding=1)
    self.layer1 = self._make_layer()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)


class RetinaNet(BaseDetectionModel):
  def __init__(self,
               name='RetinaNet',
               backbone=RetinaNetBackbone,
               fpn=None,
               classifier=None,
               regression=None):
    super(RetinaNet, self).__init__(backbone=backbone,
                                    fpn=fpn,
                                    classifier=classifier,
                                    regression=regression)
    self.name = name

  def forward(self, x):
    feature_map_backbone = self.backbone(x)
    feature_map_fpn = self.fpn(feature_map_backbone)
    cls = self.classifier(feature_map_fpn)
    reg = self.regression(feature_map_fpn)
    return cls, reg
