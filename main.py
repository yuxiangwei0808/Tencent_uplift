import os
import argparse
import time
import yaml
import random
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
import mlflow

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import torch.backends.cudnn as cudnn

from data_loader import get_data, create_folds, get_data_public
from metrics import uplift_at_k, weighted_average_uplift, metrics_mt
from sklift.metrics import uplift_auc_score, qini_auc_score
from utils import *


@torch.no_grad()
def valid(model, valid_dataloader, device, epoch, reduction):
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
        if valid_label.shape[1] > len(target_task):
            label_list = valid_label.to(device)[:, :-1]  # exclude pre30_logindays
        else:
            label_list = valid_label.to(device)

        with autocast():
            if 'efin' in args.model_name:
                _, _, _, _, _, uplift = model(feature_list, is_treat)
            elif any([x in args.model_name for x in ['dragonnet', 'crfnet', 't_learner', 'flextenet', 'tarnet']]):
                y0, y1, _, _ = model(feature_list)
                uplift = y1 - y0
            elif 'mtmt' in args.model_name:
                _, _, uplift, uplift_s = model(feature_list, is_treat)
            elif 'euen' in args.model_name:
                _, _, _, _, _, uplift = model(feature_list)
            elif 'descn' in args.model_name:
                _, _, _, _, _, _, _, _, _, p_h1, p_h0, _ = model(feature_list)
                uplift = p_h1 - p_h0
            elif 'snet' in args.model_name or 's_learner' in args.model_name:
                y0, y1, _, _ = model(feature_list, is_treat)
                uplift = y1 - y0
            else:
                raise NotImplementedError
        if uplift_s != None:
            uplift = uplift + uplift_s
            
        uplift = uplift.squeeze()
        predictions.extend(uplift.squeeze().cpu().detach())
        true_labels.extend(label_list.squeeze().cpu().detach().numpy())
        is_treatment.extend(is_treat.squeeze().cpu().detach().numpy())

        del feature_list, is_treat, label_list, uplift

    true_labels = np.array(true_labels)
    # predictions = torch.tensor(predictions)
    is_treatment = np.array(is_treatment)
    
    # pred_prob = torch.sigmoid(predictions)
    # roc_auc = roc_auc_score(true_labels, pred_prob)
    # pr_auc  = average_precision_score(true_labels, pred_prob)
    predictions = np.array(predictions)

    u_at_k, qini_coef, uplift_auc, wau = [], [], [], []

    if args.mtask:
        for i in range(true_labels.shape[1]):
            u_at_k.append(metrics_mt(uplift_at_k, true_labels[:, i], predictions, is_treatment, m_treat=len(target_treatment) > 1, reduce=reduction))
            qini_coef.append(metrics_mt(qini_auc_score, true_labels[:, i], predictions, is_treatment, m_treat=len(target_treatment) > 1, reduce=reduction))
            uplift_auc.append(metrics_mt(uplift_auc_score, true_labels[:, i], predictions, is_treatment, m_treat=len(target_treatment) > 1, reduce=reduction))
            wau.append(metrics_mt(weighted_average_uplift, true_labels[:, i], predictions, is_treatment, m_treat=len(target_treatment) > 1, reduce=reduction))
    else:
        u_at_k = metrics_mt(uplift_at_k, true_labels, predictions, is_treatment, m_treat=len(target_treatment) > 1, reduce=reduction)
        qini_coef = metrics_mt(qini_auc_score, true_labels, predictions, is_treatment, m_treat=len(target_treatment) > 1, reduce=reduction)
        uplift_auc = metrics_mt(uplift_auc_score, true_labels, predictions, is_treatment, m_treat=len(target_treatment) > 1, reduce=reduction)
        wau = metrics_mt(weighted_average_uplift, true_labels, predictions, is_treatment, m_treat=len(target_treatment) > 1, reduce=reduction)

    if isinstance(qini_coef, list): # multi-task
        if isinstance(qini_coef, tuple):  # multi-treatment multi-task
            valid_result = [[{'QINI': qini, 'AUUC': up, 'WAU': w, 'u_at_k': uk}] for (x, y, z, w) in zip(qini_coef, uplift_auc, wau, u_at_k) for qini, up, w, uk in zip(x, y, z, w)]
        else:
            valid_result = [{'QINI': qini, 'AUUC': up, 'WAU': w, 'u_at_k': uk} for qini, up, w, uk in zip(qini_coef, uplift_auc, wau, u_at_k)]
    elif isinstance(qini_coef, tuple):  # multi-treatment
        valid_result = [{'QINI': qini, 'AUUC': up, 'WAU': w, 'u_at_k': uk} for qini, up, w, uk in zip(qini_coef, uplift_auc, wau, u_at_k)]
    else:
        valid_result = {'QINI': qini_coef, 'AUUC': uplift_auc, 'WAU': wau, 'u_at_k': u_at_k}
    
    logger.info("Valid results: {} of epoch {}".format(valid_result, epoch))
    return valid_result, true_labels, predictions, is_treatment


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(local_rank, train_files, test_files, fold_idx):    
    start_epoch = 0
    best_valid_metrics = None
    result_early_stop = 0
    batch_size = 3840 * 4
    lamb = 1e-3
    learning_rate = 0.001
    metric_names = ['QINI', 'u_at_k']
    treat_names = ['5ai', '9ai']
    save_name = treat_names if args.data_type == 'warmtype' else ['login', 'diff']
    if torch.cuda.is_available():
        device = f'cuda:{args.local_rank}'
    else:
        device = 'cpu'
    model, model_kwargs = get_model(name=args.model_name, task_name=target_task)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank) if dist.is_initialized() else model
    
    ckpt_path = f"checkpoints/{args.data_type}/{args.norm_type}/{args.model_name}/{args.model_name}_{fold_idx}_"
    pred_path = f"predictions/{args.data_type}/{args.norm_type}/{args.model_name}/train/{args.model_name}_{fold_idx}_"
    
    if args.enable_dist:
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    setup_seed(seed)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=lamb)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    scaler = GradScaler()
    
    if args.resume:
        checkpoint = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1 
        best_valid_metrics = checkpoint['']

    logger.info(f'{args.model_name}: Rank {local_rank} Start Training') 
    for epoch in range(start_epoch, num_epoch):
        train_dataloader, valid_dataloader = get_data(train_files, test_files, feature_group=None, batch_size=batch_size, dist=dist.is_initialized(), target_treatment=target_treatment, target_task=target_task)
        # train_dataloader, valid_dataloader = get_data_public(batch_size, fold_idx, 'data/data_criteo.npz')
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

            if 'dragonnet' in args.model_name or not args.enable_amp:  # unsafe bce for dragonnet, which does not allow autocast
                loss = model.calculate_loss(feature_list, is_treat, label_list)
            else:
                with autocast():
                    if 'mtmt' in args.model_name:
                        loss = model.calculate_loss(feature_list, is_treat, label_list, target_task, reduction='mean')
                    else:
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
        valid_metrics, true_labels, predictions, treatment = valid(model, valid_dataloader, device, epoch, reduction)
        


        if best_valid_metrics is None:
            if isinstance(valid_metrics, list):
                if isinstance(valid_metrics[0], list):  # multi-treatment multi-task
                    best_valid_metrics = [[{'QINI': -10, 'AUUC': -10, 'WAU': -10, 'u_at_k': -10}] for _ in range(len(valid_metrics)) for _ in range(len(valid_metrics[0]))]
                else:  # multi-treatment / multi-task
                    best_valid_metrics = [{'QINI': -10, 'AUUC': -10, 'WAU': -10, 'u_at_k': -10} for _ in range(len(valid_metrics))]
            else:
                best_valid_metrics = {'QINI': -10, 'AUUC': -10, 'WAU': -10, 'u_at_k': -10}
        
        if isinstance(valid_metrics, list):
            if isinstance(valid_metrics[0], list):
                for idx_t in range(len(valid_metrics)):
                    for i in range(len(valid_metrics[idx_t])):
                        best_valid_metrics[idx_t][i], is_early_stop, result_early_stop = save_best(valid_metrics[idx_t][i], best_valid_metrics[idx_t][i], metric_names,
                                                                            model, optimizer, scaler, ckpt_path + f'{treat_names[i]}_{target_task[idx_t]}_', epoch, tr_loss, tr_steps,
                                                                            true_labels, predictions, treatment, pred_path + f'{treat_names[i]}_{target_task[idx_t]}_', result_early_stop)
            else:
                for i in range(len(valid_metrics)):
                    best_valid_metrics[i], is_early_stop, result_early_stop = save_best(valid_metrics[i], best_valid_metrics[i], metric_names,
                                                                         model, optimizer, scaler, ckpt_path + f'{save_name[i]}_', epoch, tr_loss, tr_steps,
                                                                         true_labels, predictions, treatment, pred_path + f'{save_name[i]}_', result_early_stop)   
        else:
            best_valid_metrics, is_early_stop, result_early_stop = save_best(valid_metrics, best_valid_metrics, metric_names,
                                                                            model, optimizer, scaler, ckpt_path, epoch, tr_loss, tr_steps,
                                                                            true_labels, predictions, treatment, pred_path, result_early_stop)
            
        if is_early_stop:
            result_early_stop += 1

            if result_early_stop > 5:
                break

    torch.distributed.destroy_process_group() if dist.is_initialized() else ...
    return best_valid_metrics


if __name__ == "__main__":
    seed = 114514
    num_epoch = 50
    cudnn.benchmark = True
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable_dist', action='store_true', default=False)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--fold_ids', type=int, nargs='+', help='train the given folds')
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--resume', action='store_true', default=False, help='resume training from checkpoint')
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--norm_type', type=str, default='zscore', help='normalization method for the original data')
    parser.add_argument('--model_name', type=str, default='mtmt')
    parser.add_argument('--enable_mlflow', action='store_true', default=False)
    parser.add_argument('--enable_amp', action='store_true', default=False)
    parser.add_argument('--data_type', type=str, default='full', choices=['full', 'highactive', 'midactive', 'lowactive', 'backflow', 'warmtype'], help='all data or a subset of data')
    parser.add_argument('--mtask', action='store_true', default=False)
    parser.add_argument('--note', type=str, help='additional notes')
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
        mlflow.log_artifact('./main.py', artifact_path='python_files')
        mlflow.log_artifact('./utils.py', artifact_path='python_files')
        mlflow.log_artifact('./data_loader.py', artifact_path='python_files')

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
    elif args.data_type == 'warmtype':
        file_paths = [f'data/train_test_data/traindata_warmtype_240119_240411_{args.norm_type}/dataset_{i}.hdf5' for i in range(10)]
    else:
        raise NotImplementedError
    
    if args.data_type == 'combine0':
        folds_back = create_folds([f'data/traindata_backflow_240119_240411_{args.norm_type}/dataset_{i}.hdf5' for i in range(10)])
        folds_low  = create_folds([f'data/traindata_lowactive_240119_240411_{args.norm_type}/dataset_{i}.hdf5' for i in range(10)])
        folds_mid  = create_folds([f'data/traindata_midactive_240119_240411_{args.norm_type}/dataset_{i}.hdf5' for i in range(10)])
        folds = [[t0 + t1 + t2 for t0, t1, t2 in zip(x, y, z)] for x, y, z in zip(folds_back, folds_low, folds_mid)]
    else:
        folds = create_folds(file_paths, n_folds=args.num_folds)
        
    ave_best_valid_metrics = None
    target_treatment = ['treatment_next_iswarm', 'treatment_next_is_9aiwarmround'] if args.data_type == 'warmtype' else ['treatment_next_iswarm']
    target_task = ['label_nextday_login'] if not args.mtask else ['label_nextday_login', 'label_login_days_diff']
    args.data_type = 'mtask' if args.mtask else args.data_type
    reduction = 'mean'
    
    fold_enumerator = enumerate(folds)
    fold_run = 0

    for fold_idx, (train_files, test_files) in fold_enumerator:
        if args.fold_ids is not None and fold_idx not in args.fold_ids:
            continue
        logger.info("Fold {} start".format(fold_idx))        
        
        if args.num_folds == 1:  # use all data to train and use the random test data
            args.data_type = 'all'
            logger.info('use all data to train the model and use random data to select the best epoch')
            test_files = ['data/train_test_data/testdata_240412_240611_zscore/dataset_random_0.hdf5']
        
        best_valid_metrics = train(local_rank, train_files, test_files, fold_idx)
        print(f'best metrics for fold {fold_idx}: {best_valid_metrics}')
        fold_run += 1
        
        if ave_best_valid_metrics is None:
            if isinstance(best_valid_metrics, list):
                if isinstance(best_valid_metrics[0], list):  # multi-treatment multi-task
                    ave_best_valid_metrics = [[{'QINI': 0, 'AUUC': 0, 'WAU': 0, 'u_at_k': -10}] for _ in range(len(best_valid_metrics)) for _ in range(len(best_valid_metrics[0]))]
                else:  # multi-treatment
                    ave_best_valid_metrics = [{'QINI': 0, 'AUUC': 0, 'WAU': 0, 'u_at_k': -10} for _ in range(len(best_valid_metrics))]
            else:
                ave_best_valid_metrics = {'QINI': 0., 'AUUC': 0., 'WAU': 0., 'u_at_k': 0.}
        
        if isinstance(best_valid_metrics, list):
            if isinstance(best_valid_metrics[0], list):  # multi-treatment multi-task
                for idx_t in range(len(best_valid_metrics)):
                    for i in range(len(best_valid_metrics[idx_t])):
                        ave_best_valid_metrics[idx_t][i] = {k: ave_best_valid_metrics[idx_t][i][k] + best_valid_metrics[idx_t][i][k] for k in best_valid_metrics[idx_t][i]}
            else:
                for i in range(len(best_valid_metrics)):
                    ave_best_valid_metrics[i] = {k: ave_best_valid_metrics[i][k] + best_valid_metrics[i][k] for k in best_valid_metrics[i]}
        else:
            ave_best_valid_metrics = {k: ave_best_valid_metrics[k] + best_valid_metrics[k] for k in best_valid_metrics}
        
        with open(f"predictions/{args.data_type}/{args.norm_type}/{args.model_name}/job_status.txt", 'a') as f:
            f.write(f'\nfinished fold {fold_idx}')
    
    if isinstance(best_valid_metrics, list):
        if isinstance(best_valid_metrics[0], list):  # multi-treatment multi-task
            for idx_t in range(len(best_valid_metrics)):
                for i in range(len(best_valid_metrics[idx_t])):
                    ave_best_valid_metrics[idx_t][i] = {k: v / fold_run for k, v in ave_best_valid_metrics[idx_t][i].items()}
        else:
            for i in range(len(best_valid_metrics)):
                ave_best_valid_metrics[i] = {k: v / fold_run for k, v in ave_best_valid_metrics[i].items()}
    else:
        ave_best_valid_metrics = {k: v / fold_run for k, v in ave_best_valid_metrics.items()}
    print(f'average best: {ave_best_valid_metrics}')
