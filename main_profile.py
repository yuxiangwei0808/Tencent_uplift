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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import torch.backends.cudnn as cudnn

from data_loader import get_data, create_folds
from metrics import uplift_at_k, weighted_average_uplift
from models.efin import EFIN


def save_model(model, optimizer, path, epoch, loss, metric_name, metric_value):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        metric_name: metric_value
    }
    torch.save(checkpoint, path)
    

class WrapperModel(nn.Module):
    def __init__(self, model):
        super(WrapperModel, self).__init__()
        self.model = model

    def forward(self, feature_list, is_treat, label_list):
        final_output = self.model.calculate_loss(feature_list, is_treat, label_list)
        return final_output


def valid(model, valid_dataloader, device, metric):
    logger.info('Start Verifying')
    model.eval()
    predictions = []
    true_labels = []
    is_treatment = []

    for step, (X, T, valid_label) in enumerate(tqdm(valid_dataloader)):
        feature_list = X.to(device)
        is_treat = T.to(device)
        label_list = valid_label.to(device)
        with autocast():
            _, _, _, _, _, u_tau = model.model.forward(feature_list, is_treat)
        uplift = u_tau.squeeze()

        predictions.extend(uplift.detach().cpu().numpy())
        true_labels.extend(label_list.detach().cpu().numpy())
        is_treatment.extend(is_treat.detach().cpu().numpy())

    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    is_treatment = np.array(is_treatment)

    u_at_k = uplift_at_k(true_labels, predictions, is_treatment, strategy='overall', k=0.3)
    qini_coef = qini_auc_score(true_labels, predictions, is_treatment)
    uplift_auc = uplift_auc_score(true_labels, predictions, is_treatment)
    wau = weighted_average_uplift(true_labels, predictions, is_treatment, strategy='overall')

    valid_result = [u_at_k, qini_coef, uplift_auc, wau]

    if metric == "AUUC":
        valid_metric = uplift_auc
    elif metric == "QINI":
        valid_metric = qini_coef
    elif metric == 'WAU':
        valid_metric = wau
    else:
        valid_metric = u_at_k
    logger.info("Valid results: {}".format(valid_result))
    return valid_metric, valid_result, true_labels, predictions, is_treatment


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(local_rank, train_files, test_files):
    batch_size = 3840
    rank = 96
    rank2 = 96
    lamb = 1e-3
    learning_rate = 0.001
    ckpt_path = f'checkpoints/efin/efin_{rank}_{rank2}_{lamb}_{fold_idx}.pth'
    
    if args.enable_dist:
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    train_dataloader, valid_dataloader = get_data(train_files, test_files, target_treatment=None, target_task=None, batch_size=batch_size, dist=dist.is_initialized())

    setup_seed(seed)

    model = EFIN(input_dim=621, hc_dim=rank, hu_dim=rank2, is_self=False, act_type="elu")
    model = WrapperModel(model).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank) if dist.is_initialized() else model

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=lamb)
    scaler = GradScaler()

    best_valid_metric = 0
    result_early_stop = 0
    logger.info(f'EFIN: Rank {local_rank} Start Training') 

    # Start profiling
    if local_rank == 0:
        torch.cuda.cudart().cudaProfilerStart()

    for epoch in range(num_epoch):
        tr_loss = 0
        tr_steps = 0
        logger.info("Training Epoch: {}/{}".format(epoch + 1, int(num_epoch)))
        if dist.is_initialized():
            train_dataloader.sampler.set_epoch(epoch)
        torch.cuda.nvtx.range_push("loading data")
        for step, (X, T, label) in enumerate(tqdm(train_dataloader)):
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("step_{}".format(step))
            tr_steps += 1

            torch.cuda.nvtx.range_push("data to cuda")
            feature_list = X.to(device)
            is_treat = T.to(device)
            label_list = label.to(device)
            torch.cuda.nvtx.range_pop()

            model.train()
            optimizer.zero_grad()

            torch.cuda.nvtx.range_push("forward")
            with autocast():
                loss = model(feature_list, is_treat, label_list)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("backward")
            scaler.scale(loss).backward()
            torch.cuda.nvtx.range_pop()
            
            torch.cuda.nvtx.range_push("opt step")
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.nvtx.range_pop()

            tr_loss += loss.item()
                
            torch.cuda.nvtx.range_pop()
            
            torch.cuda.nvtx.range_push("loading data")
            
            if step > 300:
                break
            
        torch.cuda.cudart().cudaProfilerStop()

        # if local_rank == 0:
        logger.info("Epoch loss: {}, Avg loss: {}".format(tr_loss, tr_loss / tr_steps))

    return best_valid_metric


if __name__ == "__main__":
    seed = 114514
    num_epoch = 1
    metric = 'QINI'
    cudnn.benchmark = True
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable_dist', action='store_true', default=False)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--reversed', action='store_true', default=False, help='iter the folds reversely')
    args = parser.parse_args()

    setup_seed(seed)

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    local_rank = args.local_rank
    world_size = args.world_size
    
    file_paths = [f'data/train_data_240119_240411_zscore/dataset_{i}.hdf5' for i in range(10)]
    folds = create_folds(file_paths, n_folds=5)
    
    ave_best_valid_metric = 0.
    enumerate_fold = enumerate(folds) if not args.reversed else reversed(list(enumerate(folds)))
    
    for fold_idx, (train_files, test_files) in enumerate_fold:
        logger.info("Fold {} start".format(fold_idx))
        train_dataloader, valid_dataloader = get_data(train_files, test_files, target_treatment=None, target_task=None, batch_size=3840, dist=dist.is_initialized())
        
        best_valid_metric = train(local_rank, train_files, test_files)
        print(f'best metrics for fold {fold_idx}: {best_valid_metric}')
        ave_best_valid_metric += best_valid_metric
    
    print(f'average best {metric}: {ave_best_valid_metric}')
