import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mt_weighting import *
from models.mmoe import MMOE
from models.base_models import *
from models.mtmt import MTMT

weighting = DWA
architecture = MMOE

class MTLmodel(architecture, weighting):
    def __init__(self, **kwargs):
        super(MTLmodel, self).__init__(**kwargs)
        self.init_param()

enc = MTLmodel(task_names=['label_nextday_login', 'label_login_days_diff'], encoder_class=resnet18, num_experts=4, rep_grad=False,
                 in_feat=626, enc_kwargs={'all': {'hidden_dim': 16, 'out_dim': None, 'drop': 0.2}}).cuda()

model = MTMT(user_feat_enc=enc, treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16), task_names=['label_nextday_login', 'label_login_days_diff'],
                 t_dim=1, u_dim=128, tu_dim=256, num_cls=[1, 1]).cuda()

x = torch.randn(2, 626).cuda()
treat = torch.ones(2).cuda()
y = torch.ones(2).cuda()

u_logit, tu_tau, tu_logit, tu_logit_s = model(x, treat)

model.user_enc.epoch = 1
# loss = sum([F.mse_loss(u_logit[t].squeeze(), y) for t in u_logit])
loss = torch.zeros(2, device=x.device)
loss[0] = F.mse_loss(u_logit['label_nextday_login'].squeeze(), y)
loss[1] = F.mse_loss(u_logit['label_login_days_diff'].squeeze(), y)

print(loss)
model.user_enc.backward(loss)