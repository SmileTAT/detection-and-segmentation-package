import sys
sys.path.append('../../')
import numpy as np
import torch
from segmentation_package.deeplab.deeplab_v3 import DeepLabV3


X = np.random.rand(32, 3, 224, 224)
y = np.random.randint(low=0, high=20, size=32)

model = DeepLabV3()
model = model.cuda(device='cuda')
pred = model(torch.tensor(X, dtype=torch.float32).cuda())
print(pred.size())

