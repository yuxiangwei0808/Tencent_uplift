import torch
import torch.nn as nn
from models.mtmt import *

x = torch.randn(3840, 622)
t = torch.ones(1)
user_feat_enc_hidden_dim=16
model = MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=38), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=16), task_names=['nextday_login'],
                 num_treats=1, t_dim=1, u_dim=128, tu_dim=256)

print(model(x, torch.zeros(3840)))