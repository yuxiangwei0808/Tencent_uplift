import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import argparse
from tqdm import tqdm
from sklift.metrics import uplift_auc_score, qini_auc_score, uplift_by_percentile
from metrics import uplift_at_k, weighted_average_uplift, metrics_mt

from data_loader import get_data, create_folds
from utils import *
from models.efin import EFIN
from models.dragonnet import DragonNet


@torch.no_grad()
def valid(model, valid_dataloader, device, num_org_feat, reduction=None):
    model.eval()
    predictions = []
    true_labels = []
    is_treatment = []
    add_features = []

    for step, (X, T, valid_label) in enumerate(tqdm(valid_dataloader)):
        if isinstance(X, list):
            feature_list = [X[0][:, :-num_org_feat], X[1]]
            feature_list = [x.to(device) for x in feature_list]
            add_features.extend(X[0][:, -num_org_feat:].numpy())
        else:
            feature_list = X[:, :-num_org_feat].to(device)
            add_features.extend(X[:, -num_org_feat:].numpy())
        is_treat = T.to(device)
        label_list = valid_label.to(device)
        if 'efin' in args.model_name:
            _, _, _, _, _, uplift = model(feature_list, is_treat)
        elif 'dragonnet' in args.model_name:
            y0, y1, _, _ = model(feature_list)
            uplift = y1 - y0
        elif 'mtmt' in args.model_name:
            # y0, _, y1 = model(feature_list, is_treat)
            # uplift = y1 - y0['nextday_login']
            _, _, uplift = model(feature_list, is_treat)
        else:
            raise NotImplementedError
        uplift = uplift.squeeze()

        predictions.extend(uplift.detach().cpu())
        true_labels.extend(label_list.detach().cpu().numpy())
        is_treatment.extend(is_treat.detach().cpu().numpy())

    true_labels = np.array(true_labels)
    # predictions = torch.tensor(predictions)
    is_treatment = np.array(is_treatment)
    add_features = np.array(add_features)
    
    # pred_prob = torch.sigmoid(predictions)
    # roc_auc = roc_auc_score(true_labels, pred_prob)
    # pr_auc  = average_precision_score(true_labels, pred_prob)
    predictions = np.array(predictions)

    if len(target_treatment) > 1:
        u_at_k = metrics_mt(uplift_at_k, true_labels, predictions, is_treatment, reduce=reduction)
        qini_coef = metrics_mt(qini_auc_score, true_labels, predictions, is_treatment, reduce=reduction)
        uplift_auc = metrics_mt(uplift_auc_score, true_labels, predictions, is_treatment, reduce=reduction)
        wau = metrics_mt(weighted_average_uplift, true_labels, predictions, is_treatment, reduce=reduction)
    else:
        u_at_k = uplift_at_k(true_labels, predictions, is_treatment, strategy='overall', k=0.3)
        qini_coef = qini_auc_score(true_labels, predictions, is_treatment)
        uplift_auc = uplift_auc_score(true_labels, predictions, is_treatment)
        wau = weighted_average_uplift(true_labels, predictions, is_treatment, strategy='overall')

    if isinstance(qini_coef, tuple):
        valid_result = [{'QINI': qini, 'AUUC': up, 'WAU': w, 'u_at_k': uk} for qini, up, w, uk in zip(qini_coef, uplift_auc, wau, u_at_k)]
    else:
        valid_result = {'QINI': qini_coef, 'AUUC': uplift_auc, 'WAU': wau, 'u_at_k': u_at_k}
    return valid_result, true_labels, predictions, is_treatment, add_features


def main(args):    
    cudnn.benchmark = True
    treat_names = ['5AI', '9AI']
        
    model, model_kawrgs = get_model(name=args.model_name)
    model = model.to(device)

    model = torch.compile(model)

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print('valid metrics: {} of {}'.format(checkpoint['metric'], args.ckpt_path))
    
    valid_metrics, true_labels, predictions, treatment, add_features = valid(model, valid_dataloader, device, len(org_feat_idx))
    print('test metrics: {} of {}'.format(valid_metrics, args.ckpt_path))
    
    saving_path = args.ckpt_path.split('/')[-1][:-4]
    saving_path = f'predictions/{args.data_type}/{args.norm_type}/{args.model_name}/test/{saving_path}'

    if isinstance(valid_metrics, list):
        for i in range(len(valid_metrics)):
            save_predictions(true_labels, predictions, treatment, valid_metrics[i], '', saving_path + f'{treat_names[i]}_')
    else:
        save_predictions(true_labels, predictions, treatment, valid_metrics[i], '', saving_path)
    
    return add_features
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--norm_type', type=str, default='zscore', help='normalization method for the original data')
    parser.add_argument('--model_name', type=str, default='efin')
    parser.add_argument('--data_type', type=str, default='full', choices=['full', 'highactive', 'midactive', 'lowactive', 'backflow'], help='all data or a subset of data')
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f'cuda:{args.local_rank}')
    
    batch_size = 3840 * 16

    file_path = [f'data/train_test_data/testdata_240412_240611_{args.norm_type}/dataset_{args.data_type}_0.hdf5']

    with open('data/train_test_data/OUT_COLUMN_new', 'r') as f:
        labels = f.readlines()

    labels = [x.strip('\n') for x in labels]
    org_feat_idx = [i for i in range(len(labels)) if 'origin' in labels[i]]
    target_treatment = ['treatment_next_iswarm', 'treatment_next_is_9aiwarmround']
    
    train_dataloader, valid_dataloader = get_data([*file_path], [*file_path], feature_group=None, batch_size=batch_size, addition_feat=org_feat_idx, target_treatment=target_treatment)
    
    for metric in ['QINI', 'u_at_k']:
        ckpt_paths = os.listdir(f'checkpoints/{args.data_type}/{args.norm_type}/{args.model_name}')
        ckpt_paths = [x for x in ckpt_paths if metric in x]
        
        for path in ckpt_paths:
            args.ckpt_path = os.path.join(f'checkpoints/{args.data_type}/{args.norm_type}/{args.model_name}', path)
            assert args.norm_type in args.ckpt_path
            add_features = main(args)
        
    np.savez_compressed(f'predictions/{args.data_type}/{args.norm_type}/{args.model_name}/test/add_features', feature=add_features)