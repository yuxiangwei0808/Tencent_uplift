import torch
import numpy as np
import os

from models.efin import EFIN
from models.dragonnet import DragonNet
from models.mtmt import *


def check_and_make_dir(path):
    dir = path.split('/')[:-1]
    dir = os.path.join(*dir)
    if not os.path.isdir(dir):
        os.makedirs(os.path.join(dir))


def save_model(model, optimizer, scaler, path, epoch, loss, metric_name, metrics):
    metric_name = '' if metric_name is None else metric_name
    checkpoint = dict({
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        # 'scaler_state_dict': scaler.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'metric': metrics})
    check_and_make_dir(path)
    torch.save(checkpoint, path + metric_name + '.pth')


def save_predictions(target, pred, treat, valid_metrics, metric_name, path, feature=None):
    check_and_make_dir(path)
    np.savez_compressed(path + metric_name, target=target, pred=pred, treat=treat, feature=feature, **valid_metrics)


def get_model(name, model_kwargs=None):
    if 'efin' in name:
        if model_kwargs is None:
            model_kwargs = {'input_dim': 622, 'hc_dim': 1024, 'hu_dim': 384, 'is_self': False, 'act_type': 'elu'}
        return EFIN(**model_kwargs), model_kwargs
    elif 'dragonnet' in name:
        if model_kwargs is None:
            model_kwargs = {'input_dim': 622, 'shared_hidden': 512, 'outcome_hidden': 256, 'is_regularized': False}
        return DragonNet(**model_kwargs), model_kwargs
    elif 'mtmt' in name:
        if model_kwargs is None:
            model_kwargs = {'name': 'mtmt_res_emb_v0_MulAttn0', 't_dim': 1, 'u_dim': 128, 'tu_dim':256}
        return mtmt_res_emb_v0_MulAttn0(), model_kwargs
    else:
        raise NotImplementedError
