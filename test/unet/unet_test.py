import sys
sys.path.append('../../')
import numpy as np
import torch
from segmentation_package.unet.unet import UNet


X = np.random.rand(1, 3, 224, 224)
y = np.random.randint(low=0, high=20, size=1)

model = UNet()
model = model.cuda(device='cuda')
pred = model(torch.tensor(X, dtype=torch.float32).cuda())
print(pred.size())

