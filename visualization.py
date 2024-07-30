import argparse
import numpy as np
import torch
from scipy.interpolate import interp1d
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt

from captum.attr import LRP, Saliency
from captum.attr import IntegratedGradients, GradientShap, FeatureAblation
from captum.attr import visualization as viz

from data_loader import get_data, create_folds
from utils import *
from models.efin import EFIN
from models.model_hub import *


@torch.no_grad()
def main_attn():
    attrs00, attrs01, attrs10, attrs11 = [], [], [], []

    for i in range(5):
        checkpoint = torch.load(ckpt_paths[i], map_location=device)['model_state_dict']
        state_dict = OrderedDict()
        for k, v in checkpoint.items():
            state_dict[k[10:]] = v
        model.load_state_dict(state_dict)
        model.eval()

        for X, T, valid_label in valid_dataloader:
            X, T = X.to(device), T.to(device)
            valid_label = valid_label[:, 0]
            _, _, _, _, attrs = model(X, T)
            T = T.cpu()

            weight = torch.pinverse(model.treatment_enc.weight)
            attrs = torch.matmul(attrs.permute(0, 2, 1), weight).cpu()

            attrs00.append(attrs[(T == 0) & (valid_label == 0)].mean(0, keepdim=True))
            attrs01.append(attrs[(T == 0) & (valid_label == 1)].mean(0, keepdim=True))
            attrs10.append(attrs[(T == 1) & (valid_label == 0)].mean(0, keepdim=True))
            attrs11.append(attrs[(T == 1) & (valid_label == 1)].mean(0, keepdim=True))

    attrs00, attrs01, attrs10, attrs11 = torch.cat(attrs00, 0).mean(0), torch.cat(attrs01, 0).mean(0), torch.cat(attrs10, 0).mean(0), torch.cat(attrs11, 0).mean(0)
    np.savez_compressed('attn', uTuL=attrs00, uTL=attrs01, TuL=attrs10, TL=attrs11)
    

@torch.no_grad()
def main_grad():
    attrs00, attrs01, attrs10, attrs11 = [], [], [], []

    for i in range(5):
        checkpoint = torch.load(ckpt_paths[i], map_location=device)['model_state_dict']
        state_dict = OrderedDict()
        for k, v in checkpoint.items():
            state_dict[k[10:]] = v

        model.load_state_dict(state_dict)
        model.eval()

        attributer = IntegratedGradients(model)
        # attributer = Saliency(model)
        # attributer = FeatureAblation(model)
        # attributer = LRP(model)
        print(attributer)

        for X, T, valid_label in tqdm(valid_dataloader):
            X, T = X.to(device), T.to(device)
            valid_label = valid_label[:, 0]
            input = torch.cat((X, T.to(torch.float32).unsqueeze(1)), dim=1)
            attrs = attributer.attribute(input).cpu()
            T = T.cpu()
            attrs00.append(attrs[(T == 0) & (valid_label == 0)].mean(0, keepdim=True))
            attrs01.append(attrs[(T == 0) & (valid_label == 1)].mean(0, keepdim=True))
            attrs10.append(attrs[(T == 1) & (valid_label == 0)].mean(0, keepdim=True))
            attrs11.append(attrs[(T == 1) & (valid_label == 1)].mean(0, keepdim=True))
    attrs00, attrs01, attrs10, attrs11 = torch.cat(attrs00, 0).mean(0), torch.cat(attrs01, 0).mean(0), torch.cat(attrs10, 0).mean(0), torch.cat(attrs11, 0).mean(0)
    np.savez_compressed('attrs_ig', uTuL=attrs00, uTL=attrs01, TuL=attrs10, TL=attrs11)


def interpolate(data, target_size):
    new_data = np.zeros((target_size, data.shape[1]))
    for i in range(data.shape[1]):
        f = interp1d(np.linspace(0, 1, len(data)), data[:, i], kind='nearest')
        new_data[:, i] = f(np.linspace(0, 1, target_size))
    return new_data


def process_output():
    with open('data/train_test_data/OUT_COLUMN_new', 'r') as f:
        labels = f.readlines()
    labels = [x.strip('\n') for x in labels]
    feature_index = [idx for idx, elem in enumerate(labels) if elem[:3] == 'fea' and 'login_days' not in elem]
    feature_index = list(filter(lambda i: i not in list(range(537, 607)), feature_index))
    labels = [labels[i] for i in feature_index]
    labels.append('treatment')

    target = np.load('attrs_ig.npz')
    uTuL, uTL, TuL, TL = target['uTuL'], target['uTL'], target['TuL'], target['TL']

    # uTuL = interpolate(uTuL, len(labels) - 1)
    # uTL = interpolate(uTL, len(labels) - 1)
    # TuL = interpolate(TuL, len(labels) - 1)
    # TL = interpolate(TL, len(labels) - 1)

    data = {labels[i]: [row[i] for row in [uTuL, uTL, TuL, TL]] for i in range(len(labels))}
    df = pd.DataFrame(data, index=['noTreatnoLogin', 'noTreatLogin', 'TreatnoLogin', 'TreatLogin'])
    df.to_csv('attrs_ig.csv')


def analysis():
    with open('data/train_test_data/OUT_COLUMN_new', 'r') as f:
        labels = f.readlines()
    labels = [x.strip('\n') for x in labels]
    feature_index = [idx for idx, elem in enumerate(labels) if elem[:3] == 'fea' and 'login_days' not in elem]
    feature_index = list(filter(lambda i: i not in list(range(537, 607)), feature_index))
    labels = [labels[i] for i in feature_index]
    attr = pd.read_csv('attrs_lrp.csv')

    for i, name in enumerate(attr['Unnamed: 0']):
        with open('analyze_attrs_lrp.txt', 'a') as f:
            f.write(f'================= {name} ====================\n')
        s = attr.loc[i][1:]
        # for q in [0.99, 0.95, 0.9, 0.85, 0.8]:
        #     top_elem = s[s >= s.quantile(q)].sort_values(ascending=0)
        for q in [0.01, 0.05, 0.1, 0.15, 0.2]:
            top_elem = s[s <= s.quantile(q)].sort_values(ascending=0)
            with open('analyze_attrs_lrp.txt', 'a') as f:
                f.write(f'quantile: {q}: {", ".join(top_elem.index)}\n')

    # target = np.load('attn.npz')
    # uTuL, uTL, TuL, TL = target['uTuL'], target['uTL'], target['TuL'], target['TL']

    # ind = np.where(np.abs(TL>0.03))
    # ind_label = [labels[i] for i in ind[0]]
    # ind_label_treat = [ind_label[i] for i in range(len(ind_label)) if ind[1][i] == 1]
    # ind_label_control = [ind_label[i] for i in range(len(ind_label)) if ind[1][i] == 0]
    # with open('attn.txt', 'a') as f:
    #     f.write('============= treat come =================\n')
    #     f.write(f'Treat: {", ".join(ind_label_treat)}\n')
    #     f.write(f'Control: {", ".join(ind_label_control)}\n')

    # plt.figure()
    # plt.imshow(TL, cmap='viridis', aspect='auto')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.xticks([0, 1])
    # plt.yticks(ticks=np.unique(ind[0]))
    # plt.colorbar()
    # plt.savefig('attn_TL.png')

    
if __name__ == '__main__':
    analysis()
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--norm_type', type=str, default='zscore', help='normalization method for the original data')
    parser.add_argument('--model_name', type=str, default='mtmt')
    parser.add_argument('--data_type', type=str, default='full', choices=['full', 'highactive', 'midactive', 'lowactive', 'backflow', 'warmtype'], help='all data or a subset of data')
    parser.add_argument('--test_data_type', type=str, default='random')
    args = parser.parse_args()
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(device)
    batch_size = 3840 // 4
    metric = 'QINI'
    args.model_name = 'mtmt_res_emb_v0_4_0_EFIN_l1+0.2l5'

    file_path = [f'data/train_test_data/testdata_240412_240611_{args.norm_type}/dataset_{args.test_data_type}_0.hdf5']

    with open('data/train_test_data/OUT_COLUMN_new', 'r') as f:
        labels = f.readlines()

    labels = [x.strip('\n') for x in labels]
    target_treatment = ['treatment_next_iswarm', 'treatment_next_is_9aiwarmround'] if args.data_type == 'warmtype' else ['treatment_next_iswarm']
    target_task = ['label_nextday_login']
    
    train_dataloader, valid_dataloader = get_data([*file_path], [*file_path], feature_group=None, batch_size=batch_size, target_treatment=target_treatment, target_task=target_task)

    model = mtmt_res_emb_v0_4_0(combined_input=True, combined_output=True)
    model = model.to(device)
    ckpt_paths = [f'checkpoints/{args.data_type}/{args.norm_type}/{args.model_name}/{args.model_name}_{i}_{metric}.pth' for i in range(5)]

    main_grad()