import torch
from torch import nn


class BaseDetectionModel(nn.Module):
  """ ."""

  def __init__(self,
               backbone,
               fpn,
               classifier,
               regression):
    super(BaseDetectionModel, self).__init__()
    self.backbone = backbone
    self.fpn = fpn
    self.classifier = classifier
    self.regression = regression

  def reset_backbone(self, backbone):
    self.backbone = backbone

  def reset_fpn(self, fpn):
    self.fpn = fpn

  def reset_classifier(self, classifier):
    self.classifier = classifier

  def reset_regression(self, regression):
    self.regression = regression

