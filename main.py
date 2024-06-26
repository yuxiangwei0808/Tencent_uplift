import os
import argparse
import time
import yaml
import random
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklift.metrics import uplift_auc_score, qini_auc_score
from sklearn.metrics import roc_auc_score, average_precision_score
import mlflow

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import torch.backends.cudnn as cudnn

from data_loader import get_data, create_folds
from metrics import uplift_at_k, weighted_average_uplift
from utils import *


@torch.no_grad()
def valid(model, valid_dataloader, device, metric, epoch):
    logger.info('Start Verifying')
    model.eval()
    predictions = []
    true_labels = []
    is_treatment = []

    for step, (X, T, valid_label) in enumerate(tqdm(valid_dataloader)):
        if isinstance(X, list):
                feature_list = [f.to(device) for f in X]
        else:
            feature_list = X.to(device)
        is_treat = T.to(device)
        label_list = valid_label.to(device)
        with autocast():
            if 'efin' in args.model_name:
                _, _, _, _, _, uplift = model(feature_list, is_treat)
            elif 'dragonnet' in args.model_name:
                y0, y1, _, _ = model(feature_list)
                uplift = y1 - y0
            elif 'mtmt' in args.model_name:
                _, _, uplift = model(feature_list, is_treat)
            else:
                raise NotImplementedError
        uplift = uplift.squeeze()

        predictions.extend(uplift.cpu().detach())
        true_labels.extend(label_list.cpu().detach().numpy())
        is_treatment.extend(is_treat.cpu().detach().numpy())
        
        del feature_list, is_treat, label_list, uplift

    true_labels = np.array(true_labels)
    # predictions = torch.tensor(predictions)
    is_treatment = np.array(is_treatment)
    
    # pred_prob = torch.sigmoid(predictions)
    # roc_auc = roc_auc_score(true_labels, pred_prob)
    # pr_auc  = average_precision_score(true_labels, pred_prob)
    predictions = np.array(predictions)

    u_at_k = uplift_at_k(true_labels, predictions, is_treatment, strategy='overall', k=0.3)
    qini_coef = qini_auc_score(true_labels, predictions, is_treatment)
    uplift_auc = uplift_auc_score(true_labels, predictions, is_treatment)
    wau = weighted_average_uplift(true_labels, predictions, is_treatment, strategy='overall')

    # valid_result = [u_at_k, qini_coef, uplift_auc, wau, roc_auc, pr_auc]
    valid_result = [u_at_k, qini_coef, uplift_auc, wau]

    if metric == "AUUC":
        valid_metric = uplift_auc
    elif metric == "QINI":
        valid_metric = qini_coef
    elif metric == 'WAU':
        valid_metric = wau
    else:
        valid_metric = u_at_k
    logger.info("Valid results: {} of epoch {}".format(valid_result, epoch))
    return {metric: valid_metric, 'AUUC': uplift_auc, 'WAU': wau, 'u_at_k': u_at_k}, valid_result, true_labels, predictions, is_treatment


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(local_rank, train_files, test_files, fold_idx):    
    start_epoch = 0
    best_valid_metrics = {'QINI': 0., 'ROC-AUC': 0, 'PR-AUC': 0., 'AUUC': 0., 'WAU': 0., 'u_at_k': 0.}
    result_early_stop = 0
    batch_size = 3840 * 8
    lamb = 1e-3
    learning_rate = 0.001
    if torch.cuda.is_available():
        device = f'cuda:{args.local_rank}'
    else:
        device = 'cpu'
    model, model_kwargs = get_model(name=args.model_name)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank) if dist.is_initialized() else model
    
    ckpt_path = f"checkpoints/{args.data_type}/{args.norm_type}/{args.model_name}/{args.model_name}_{fold_idx}_"
    pred_path = f"predictions/{args.data_type}/{args.norm_type}/{args.model_name}/train/{args.model_name}_{fold_idx}_"
    
    if args.enable_dist:
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    setup_seed(seed)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=lamb)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    scaler = GradScaler()
    
    if args.resume:
        checkpoint = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1 
        best_valid_metrics = checkpoint[metric]

    logger.info(f'EFIN: Rank {local_rank} Start Training') 
    for epoch in range(start_epoch, num_epoch):
        train_dataloader, valid_dataloader = get_data(train_files, test_files, feature_group=1, batch_size=batch_size, dist=dist.is_initialized())
        tr_loss = 0
        tr_steps = 0
        logger.info("Training Epoch: {}/{}".format(epoch + 1, int(num_epoch)))
        if dist.is_initialized():
            train_dataloader.sampler.set_epoch(epoch)
        for step, (X, T, label) in enumerate(tqdm(train_dataloader)):
            tr_steps += 1   
            
            if isinstance(X, list):
                feature_list = [f.to(device) for f in X]
            else:
                feature_list = X.to(device)
            is_treat = T.to(device)
            label_list = label.to(device)

            model.train()
            optimizer.zero_grad()

            if 'dragonnet' in args.model_name or not args.enable_amp:  # unsage bce for dragonnet, which does not allow autocast
                loss = model.calculate_loss(feature_list, is_treat, label_list)
            else:
                with autocast():
                    loss = model.calculate_loss(feature_list, is_treat, label_list)

            if args.enable_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            tr_loss += loss.item()
        scheduler.step()

        # if local_rank == 0:
        logger.info("Epoch loss: {}, Avg loss: {}".format(tr_loss, tr_loss / tr_steps))
        
        model.eval()
        valid_metrics, _, true_labels, predictions, treatment = valid(model, valid_dataloader, device, metric, epoch)
        
        if args.enable_mlflow:
            for k, v in valid_metrics.items():
                mlflow.log_metric(k + f'_{fold_idx}', v, epoch)
            mlflow.log_metric(f'train_loss_{fold_idx}', tr_loss / tr_steps, epoch)

        is_early_stop = True
        metric_names = ['QINI', 'u_at_k']
        for metric_name in metric_names:
            if valid_metrics[metric_name] > best_valid_metrics[metric_name]:
                is_early_stop = False
                save_model(model, optimizer, scaler, ckpt_path, epoch, tr_loss / tr_steps, metric_name, valid_metrics)
                save_predictions(true_labels, predictions, treatment, valid_metrics, metric_name, pred_path)
                best_valid_metrics[metric_name] = valid_metrics[metric_name]
                result_early_stop = 0
        
        if is_early_stop:
            result_early_stop += 1

            if result_early_stop > 5:
                break

    torch.distributed.destroy_process_group() if dist.is_initialized() else ...
    return best_valid_metrics


if __name__ == "__main__":
    seed = 114514
    num_epoch = 50
    metric = 'QINI'
    cudnn.benchmark = True
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable_dist', action='store_true', default=False)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--fold_ids', nargs='+', type=int, help='train the given folds')
    parser.add_argument('--resume', action='store_true', default=False, help='resume training from checkpoint')
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--norm_type', type=str, default='zscore', help='normalization method for the original data')
    parser.add_argument('--model_name', type=str, default='efin')
    parser.add_argument('--enable_mlflow', action='store_true', default=False)
    parser.add_argument('--enable_amp', action='store_true', default=False)
    parser.add_argument('--data_type', type=str, default='full', choices=['full', 'highactive', 'midactive', 'lowactive', 'backflow', 'combine0'], help='all data or a subset of data')
    args = parser.parse_args()

    setup_seed(seed)

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    if args.enable_mlflow:
        mlflow.set_tracking_uri('')
        mlflow.set_experiment('/' + args.data_type + '/' + args.norm_type)
        mlflow.set_tag('mlflow.runName', args.model_name)
        mlflow.set_tag('model_name', args.model_name)
        mlflow.log_params(vars(args))
        mlflow.log_artifacts('./models', artifact_path='python_files')

    local_rank = args.local_rank
    world_size = args.world_size
    
    
    if args.data_type == 'full':
        file_paths = [f'data/train_test_data/traindata_240119_240411_{args.norm_type}/dataset_{i}.hdf5' for i in range(10)]
    elif args.data_type == 'backflow':
        file_paths = [f'data/train_test_data/traindata_backflow_240119_240411_{args.norm_type}/dataset_{i}.hdf5' for i in range(10)]
    elif args.data_type == 'highactive':
        file_paths = [f'data/train_test_data/traindata_highactive_240119_240411_{args.norm_type}/dataset_{i}.hdf5' for i in range(10)]
    elif args.data_type == 'lowactive':
        file_paths = [f'data/train_test_data/traindata_lowactive_240119_240411_{args.norm_type}/dataset_{i}.hdf5' for i in range(10)]
    elif args.data_type == 'midactive':
        file_paths = [f'data/train_test_data/traindata_midactive_240119_240411_{args.norm_type}/dataset_{i}.hdf5' for i in range(10)]
    elif args.data_type == 'combine0':
        ...
    else:
        raise NotImplementedError
    
    if args.data_type == 'combine0':
        folds_back = create_folds([f'data/traindata_backflow_240119_240411_{args.norm_type}/dataset_{i}.hdf5' for i in range(10)])
        folds_low  = create_folds([f'data/traindata_lowactive_240119_240411_{args.norm_type}/dataset_{i}.hdf5' for i in range(10)])
        folds_mid  = create_folds([f'data/traindata_midactive_240119_240411_{args.norm_type}/dataset_{i}.hdf5' for i in range(10)])
        folds = [[t0 + t1 + t2 for t0, t1, t2 in zip(x, y, z)] for x, y, z in zip(folds_back, folds_low, folds_mid)]
    else:
        folds = create_folds(file_paths, n_folds=5)
        
    ave_best_valid_metrics = {'QINI': 0., 'ROC-AUC': 0, 'PR-AUC': 0., 'AUUC': 0., 'WAU': 0., 'u_at_k': 0.}
    
    fold_enumerator = enumerate(folds)
    fold_run = 0

    for fold_idx, (train_files, test_files) in fold_enumerator:
        if args.fold_ids is not None and fold_idx not in args.fold_ids:
            continue
        logger.info("Fold {} start".format(fold_idx))        
        
        best_valid_metrics = train(local_rank, train_files, test_files, fold_idx)
        print(f'best metrics for fold {fold_idx}: {best_valid_metrics}')
        fold_run += 1
        
        if args.enable_mlflow:
            mlflow.log_metrics({'best_' + k: v for k, v in best_valid_metrics.items()}, step=fold_idx)
        
        ave_best_valid_metrics = {k: ave_best_valid_metrics[k] + best_valid_metrics[k] for k in best_valid_metrics}
    
    ave_best_valid_metrics = {k: v / fold_run for k, v in ave_best_valid_metrics.items()}
    print(f'average best: {ave_best_valid_metrics}')
