import torch
import torch.nn as nn
from models.mtmt import *
from models.mmoe import MMOE

x = torch.randn(1, 622).cuda()
t = torch.ones(1)

x2 = torch.randn(1, 9, 35)

model = vit_tiny_patch2_224(img_size=622, in_chans=1).cuda()
y = model(x)
print(y.shape)
# model = MMOE(encoder_class=resnet18, 
#              num_experts=4, task_names=['1'], in_feat=622, 
#              enc_kwargs={'all': {'hidden_dim': 16, 'out_dim': None}},
#              rep_grad=False)

y = model([x, x2], t)
print(y.shape)