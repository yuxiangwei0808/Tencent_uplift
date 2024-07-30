import torch
import numpy as np
import os

from models.efin import EFIN
from models.baseline import *
from models.model_hub import *
from models.mt_weighting import *


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


def get_model(name, model_kwargs=None, task_name=None):
    if 'efin' in name:
        if model_kwargs is None:
            model_kwargs = {'input_dim': 629, 'hc_dim': 96, 'hu_dim': 96, 'is_self': False, 'act_type': 'elu'}
        return EFIN(**model_kwargs), model_kwargs
    elif 'dragonnet' in name:
        if model_kwargs is None:
            model_kwargs = {'input_dim': 629, 'shared_hidden': 512, 'outcome_hidden': 256, 'is_regularized': False}
        return DragonNet(**model_kwargs), model_kwargs
    elif 'mtmt' in name:
        if model_kwargs is None:
            model_kwargs = {'name': 'mtmt_res_emb_v0_4_0', 't_dim': 1, 'u_dim': 128, 'tu_dim':256}
        return mtmt_res_emb_v0_4_0(), model_kwargs
    elif 'euen' in name:
        model_kwargs = {'input_dim': 193, 'hc_dim': 64, 'hu_dim': 64}
        return EUEN(**model_kwargs), model_kwargs
    elif 'tarnet' in name:
        return TARNET(input_dim=629), None
    elif 'crfnet' in name:
        return CFRNET(input_dim=629), None
    elif 'descn' in name:
        return DESCN(input_dim=629), None
    elif 'snet' in name:
        return SNet(input_dim=629), None
    elif 'flextenet' in name:
        return FlexTENet(input_dim=629), None
    elif 's_learner' in name:
        return SLearner(input_dim=629), None
    elif 't_learner' in name:
        return TLearner(input_dim=629), None
    else:
        raise NotImplementedError


def save_best(valid_metrics, best_valid_metrics, metric_names,
             model, optimizer, scaler, ckpt_path, epoch, tr_loss, tr_steps,
             true_labels, predictions, treatment, pred_path, result_early_stop):
    is_early_stop = True
    for metric_name in metric_names:  
        if valid_metrics[metric_name] > best_valid_metrics[metric_name]:
            is_early_stop = False
            save_model(model, optimizer, scaler, ckpt_path, epoch, tr_loss / tr_steps, metric_name, valid_metrics)
            save_predictions(true_labels, predictions, treatment, valid_metrics, metric_name, pred_path)
            best_valid_metrics[metric_name] = valid_metrics[metric_name]
            result_early_stop = 0

    return best_valid_metrics, is_early_stop, result_early_stop