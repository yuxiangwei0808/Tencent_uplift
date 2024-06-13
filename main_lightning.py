import os
import random
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklift.metrics import uplift_auc_score, qini_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, RandomSampler, DistributedSampler
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data_loader import get_data, create_folds
from metrics import uplift_at_k, weighted_average_uplift
from models.efin import EFIN


class UpliftModel(LightningModule):
    def __init__(self, input_dim, rank, rank2, learning_rate, lamb):
        super(UpliftModel, self).__init__()
        self.model = EFIN(input_dim=input_dim, hc_dim=rank, hu_dim=rank2, is_self=False, act_type="elu")
        self.learning_rate = learning_rate
        self.lamb = lamb

    def forward(self, feature_list, is_treat, label_list):
        return self.model.calculate_loss(feature_list, is_treat, label_list)

    def training_step(self, batch, batch_idx):
        X, T, label = batch
        loss = self(X, T, label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, T, valid_label = batch
        with torch.cuda.amp.autocast():
            _, _, _, _, _, u_tau = self.model.forward(X, T)
        uplift = u_tau.squeeze()
        return {'uplift': uplift, 'label': valid_label, 'is_treat': T}

    def validation_epoch_end(self, outputs):
        true_labels = np.concatenate([x['label'].cpu().numpy() for x in outputs])
        predictions = np.concatenate([x['uplift'].cpu().numpy() for x in outputs])
        is_treatment = np.concatenate([x['is_treat'].cpu().numpy() for x in outputs])

        u_at_k = uplift_at_k(true_labels, predictions, is_treatment, strategy='overall', k=0.3)
        qini_coef = qini_auc_score(true_labels, predictions, is_treatment)
        uplift_auc = uplift_auc_score(true_labels, predictions, is_treatment)
        wau = weighted_average_uplift(true_labels, predictions, is_treatment, strategy='overall')

        valid_result = [u_at_k, qini_coef, uplift_auc, wau]
        self.log('val_metric', qini_coef)
        logger.info("Valid results: {}".format(valid_result))

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.lamb)
        return optimizer


def setup_seed(seed):
    seed_everything(seed, workers=True)


def main():
    batch_size = 3840
    rank = 96
    rank2 = 96
    lamb = 1e-3
    learning_rate = 0.001
    seed = 0
    num_epoch = 20
    metric = 'QINI'
    data_path = 'data/demo'

    setup_seed(seed)
    
    file_paths = [f'data/train_data_240119_240411_zscore/dataset_{i}.hdf5' for i in range(10)]
    folds = create_folds(file_paths, n_folds=5)
    all_fold_results = []

    for fold_idx, (train_files, test_files) in enumerate(folds):
        train_dataloader, valid_dataloader = get_data(train_files, test_files, target_treatment=None, target_task=None, batch_size=batch_size, dist=torch.distributed.is_initialized())
        
        model = UpliftModel(input_dim=621, rank=rank, rank2=rank2, learning_rate=learning_rate, lamb=lamb)

        checkpoint_callback = ModelCheckpoint(
            monitor='val_metric',
            save_top_k=1,
            mode='max',
            filename=f'best-checkpoint-fold{fold_idx}'
        )

        early_stop_callback = EarlyStopping(
            monitor='val_metric',
            patience=5,
            mode='max'
        )

        trainer = Trainer(
            gpus=-1 if torch.cuda.is_available() else 0,
            max_epochs=num_epoch,
            precision=16,
            callbacks=[checkpoint_callback, early_stop_callback],
        )

        trainer.fit(model, train_dataloader, valid_dataloader)
        
        best_checkpoint = checkpoint_callback.best_model_path
        model = UpliftModel.load_from_checkpoint(best_checkpoint)
        val_result = trainer.validate(model, dataloaders=valid_dataloader)
        all_fold_results.append(val_result)

        logger.info(f"Fold {fold_idx} results: {val_result}")

    mean_results = np.mean([res[0]['val_metric'] for res in all_fold_results])
    logger.info(f"Cross-Validation Mean Result: {mean_results}")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    main()
