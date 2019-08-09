from torch import nn
import torch


class Encoder(nn.Module):
  """ Encoder for UNet. """

  def __init__(self, in_channels, out_channels):
    super(Encoder, self).__init__()
    layers = [
      nn.Conv2d(in_channels, out_channels, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, 3, padding=1),
      nn.ReLU(inplace=True),
    ]
    self.encoder = nn.Sequential(*layers)

  def forward(self, x):
    return self.encoder(x)


class Decoder(nn.Module):
  """Decoder for UNet. """

  def __init__(self, in_channels, mid_channels, out_channels):
    super(Decoder, self).__init__()
    layers = [
      nn.Conv2d(in_channels, mid_channels, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(mid_channels, out_channels, 2, 2),
      nn.ReLU(inplace=True),
    ]
    self.decoder = nn.Sequential(*layers)

  def forward(self, x):
    return self.decoder(x)


class UNet(nn.Module):
  """Build UNet for segmentation tasks.
      without batch normalization. """

  def __init__(self, num_classes=21):
    super(UNet, self).__init__()
    self.num_classes = num_classes

    # conv1
    self.encoder1 = Encoder(3, 64)
    self.pool1 = nn.MaxPool2d(2, 2)

    # conv2
    self.encoder2 = Encoder(64, 128)
    self.pool2 = nn.MaxPool2d(2, 2)

    # conv3
    self.encoder3 = Encoder(128, 256)
    self.pool3 = nn.MaxPool2d(2, 2)

    # conv4
    self.encoder4 = Encoder(256, 512)
    self.pool4 = nn.MaxPool2d(2, 2)

    # center
    self.decoder4 = Decoder(512, 1024, 512)

    # conv4d
    self.decoder3 = Decoder(1024, 512, 256)

    # conv3d
    self.decoder2 = Decoder(512, 256, 128)

    # conv2d
    self.decoder1 = Decoder(256, 128, 64)

    # final
    self.final = nn.Sequential(
      nn.Conv2d(128, 64, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 64, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, self.num_classes, 1)
    )

  def forward(self, x):
    encode1 = self.encoder1(x)
    pool1 = self.pool1(encode1)

    encode2 = self.encoder2(pool1)
    pool2 = self.pool2(encode2)

    encode3 = self.encoder3(pool2)
    pool3 = self.pool3(encode3)

    encode4 = self.encoder4(pool3)
    pool4 = self.pool4(encode4)

    decode4 = self.decoder4(pool4)

    decode3 = self.decoder3(torch.cat([decode4, encode4], dim=1))

    decode2 = self.decoder2(torch.cat([decode3, encode3], dim=1))

    decode1 = self.decoder1(torch.cat([decode2, encode2], dim=1))

    out = self.final(torch.cat([decode1, encode1], dim=1))

    return out
