import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

from metrics import *
from sklift.metrics import qini_auc_score, uplift_auc_score
from sklift.viz import plot_qini_curve, plot_uplift_by_percentile, plot_uplift_curve


def analyze_online_model():
    source = target_dir = f'results/label_transform_model_newdata/lowactive/5ai'
    print(source)
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    df = pd.read_csv('predictions/label_transform_model_newdata2/LabelTransformModel_240119_240411_rank_warmtype_allgame_test_player1_module1_all_datasize9807284_2406.csv')
    targets = np.array(df['nextday_login'])
    label = np.array(df['label_transform'])
    preds = np.array(df['pred']) * 2 -1

    treats = np.array(df['next_iswarm'])
    treat_9ai = np.array(df['next_is_9aiwarmround'])
    ind_control = treats == 0
    ind_ai = (treats == 1) & (treat_9ai == 1)
    targets, preds, treats = targets[ind_control | ind_ai], preds[ind_control | ind_ai], treats[ind_control | ind_ai]
    label = label[ind_control | ind_ai]
    print(roc_auc_score(label, preds), average_precision_score(label, preds))
    add_feats = np.array(df['pre14_login_days'])[ind_control | ind_ai]


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def calc_roc_pr(targets, preds):
    preds_bin = sigmoid(preds)
    preds_bin[preds_bin >= 0.5] = 1
    preds_bin[preds_bin < 0.5] = 0
    return roc_auc_score(targets, preds_bin), average_precision_score(targets, preds_bin)


def plot_and_save_roc_auc(predictions, targets, filename='roc_auc.png'):
    fpr, tpr, _ = roc_curve(targets, predictions)
    roc_auc = roc_auc_score(targets, predictions)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()

def plot_and_save_pr_auc(predictions, targets, filename='pr_auc.png'):
    precision, recall, _ = precision_recall_curve(targets, predictions)
    pr_auc = auc(recall, precision)
    
    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curve')
    plt.legend(loc="lower left")
    plt.savefig(filename)
    plt.close()
    
    
def plot_and_save_precision_recall_vs_threshold(predictions, targets, filename='precision_recall_vs_threshold.png'):
    precisions, recalls, thresholds = precision_recall_curve(targets, predictions)
    
    plt.figure()
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Precision/Recall')
    plt.title('Precision and Recall vs Threshold')
    plt.legend(loc="best")
    plt.savefig(filename)
    plt.close()
    

def analysis_and_plot_uplift_qini_curves(targets, predictions, treats):
    # uplift_perc, qini, auuc, u_at_k = analysis(targets, predictions, treats)
    uplift_perc = [uplift_by_percentile(t, p, tr) for t, p, tr in zip(targets, predictions, treats)]
    uplift_perc = pd.concat(uplift_perc)
    uplift_perc = uplift_perc.groupby(level=0).mean()
    print(uplift_perc)
    uplift_perc.to_csv(f'{source_dir}/uplift_perc.txt', sep='\t')

    targets, predictions, treats = np.concatenate(targets, 0), np.concatenate(predictions, 0), np.concatenate(treats, 0)
    
    plt.figure()
    sns.displot(predictions, bins=100, kde=True)
    # plt.xlim(-0.1, 0.5)
    plt.grid(True)
    plt.savefig(f'{source_dir}/u_trend.png', bbox_inches='tight') 
    
    up_per = plot_uplift_by_percentile(targets, predictions, treats, kind='bar')
    plt.savefig(f'{source_dir}/uplift_by_percentile.png', bbox_inches='tight')
    
    qini_disp = plot_qini_curve(targets, predictions, treats)
    plt.savefig(f'{source_dir}/qini.png', bbox_inches='tight')
    
    auuc_disp = plot_uplift_curve(targets, predictions, treats)
    plt.savefig(f'{source_dir}/uplift.png', bbox_inches='tight')


def analysis(targets, preds, treats):
    preds_bin = sigmoid(preds)
    preds_bin[preds_bin >= 0.5] = 1
    preds_bin[preds_bin < 0.5] = 0
    
    # print('==================== Classification Report ====================')
    # print(classification_report(targets, preds_bin))

    # print('==================== Confusion Matrix ====================')
    # print(confusion_matrix(targets, preds_bin))

    uplifts = uplift_by_percentile(targets, preds, treats)
    u_at_k = uplift_at_k(targets, preds, treats, strategy='overall', k=0.3)

    qini = qini_auc_score(targets, preds, treats)
    auuc = uplift_auc_score(targets, preds, treats)

    # plot_and_save_roc_auc(preds, targets)
    # plot_and_save_pr_auc(preds, targets)
    # plot_and_save_precision_recall_vs_threshold(preds, targets)
    
    return uplifts, qini, auuc, u_at_k


def analysis_by_logindays(targets, preds, treats, add_feats):
    # analyze the results by each login days (add_feats[:, 1])
    uplifts_percentiles, qinis, auucs, uks = [[] for _ in range(5)], [[] for _ in range(5)], [[] for _ in range(5)], [[] for _ in range(5)]
    
    if 'backflow' in source:
        ranger = np.arange(1)
    elif 'highactive' in source:
        ranger = np.arange(10, 15)
    elif 'midactive' in source:
        ranger = np.arange(5, 10)
    elif 'lowactive' in source:
        ranger = np.arange(1, 5)
    else:
        ranger = np.arange(15)
    
    for i, (t, p, tr, af) in enumerate(zip(targets, preds, treats, add_feats)):
        for day in ranger:
            idx = af[:, 1] == day
            uplifts, qini, auuc, u_at_k = analysis(t[idx], p[idx], tr[idx])
            uplifts_percentiles[i].append(uplifts)
            qinis[i].append(qini)
            auucs[i].append(auuc)
            uks[i].append(u_at_k)
    
    percentiles = list(uplifts['uplift'].keys())

    qinis, auucs, uks = np.array(qinis).mean(0), np.array(auucs).mean(0), np.array(uks).mean(0)
    u_percentiles = []
    for i in range(len(uplifts_percentiles[0])):
        u_percentiles.append(pd.concat([uplifts_percentiles[j][i] for j in range(5)]).groupby(level=0).mean())
    uplifts_percentiles = u_percentiles
            
    print('average qini: ', np.mean(qinis))
    print('average auuc: ', np.mean(auucs))
    print('average u at 0.3: ', np.mean(uks))

    with open(f'{source_dir}/results.txt', 'a') as f:
        f.write(f'\nby login days -- qini: {np.mean(qinis)}, auuc: {np.mean(auucs)}, u_0.3: {np.mean(uks)}')
    
    dfs = pd.concat(uplifts_percentiles)
    dfs = dfs.groupby(level=0).mean()
    print('average percentile: ', dfs)

    plt.figure()
    plt.plot(ranger, qinis)
    plt.grid(True)
    plt.xlabel('login days')
    plt.ylabel('QINI')
    plt.savefig(f'{source_dir}/qini_login.png')
    
    plt.figure()
    plt.plot(ranger, auucs)
    plt.grid(True)
    plt.xlabel('login days')
    plt.ylabel('AUUC')
    plt.savefig(f'{source_dir}/auuc_login.png')
    
    fig, ax = plt.subplots()
    for p in percentiles:
        ufs = [x.loc[p]['uplift'] for x in uplifts_percentiles]
        ax.plot(ranger, ufs, label=f'percentile={p}')
    ax.legend(fontsize='5')
    ax.grid(True)
    ax.set_xlabel('login days')
    ax.set_ylabel('uplifts')
    plt.savefig(f'{source_dir}/uplift_by_percentile_login.png')


def plot_qinis_auucs_by_rank(targets, preds, treats, rank_name):
    if not os.path.isdir(f'{source_dir}/qinis'):
        os.mkdir(f'{source_dir}/qinis')
    if not os.path.isdir(f'{source_dir}/auucs'):
        os.mkdir(f'{source_dir}/auucs')

    qini_disp = plot_qini_curve(targets, preds, treats)
    plt.savefig(f'{source_dir}/qinis/qini_{rank_name}.png', bbox_inches='tight')
    plt.close()
    
    auuc_disp = plot_uplift_curve(targets, preds, treats)
    plt.savefig(f'{source_dir}/auucs/auuc_{rank_name}.png', bbox_inches='tight')
    plt.close()


def analysis_by_rank(targets, preds, treats, add_feats):
    ranks = [[0], np.arange(1, 5), np.arange(5, 9), np.arange(9, 13), np.arange(13, 17), np.arange(17, 21)]
    if data_type != 'warmtype':
        rank_names = ['unrank', 'iron', 'copper', 'silver', 'gold', 'plat', 'em4', 'em3', 'em2', 'em1', 'dia4', 'dia3', 'dia2', 'dia1', 'master0']
        ranks.extend([[i] for i in range(21, 30)])
    else:
        rank_names = ['unrank', 'iron', 'copper', 'silver', 'gold', 'plat']

    uplifts_percentiles, qinis, auucs, uks = [[] for _ in range(5)], [[] for _ in range(5)], [[] for _ in range(5)], [[] for _ in range(5)]

    for i, (t, p, tr, af) in enumerate(zip(targets, preds, treats, add_feats)):
        for rank_id, rank in enumerate(ranks):
            idx = [i for i, r in enumerate(af[:, 0]) if r in rank]
            uplifts, qini, auuc, u_at_k = analysis(t[idx], p[idx], tr[idx])
            plot_qinis_auucs_by_rank(t[idx], p[idx], tr[idx], rank_names[rank_id])

            uplifts_percentiles[i].append(uplifts)
            qinis[i].append(qini)
            auucs[i].append(auuc)
            uks[i].append(u_at_k)

    qinis, auucs, uks = np.array(qinis).mean(0), np.array(auucs).mean(0), np.array(uks).mean(0)
    u_percentiles = []
    for i in range(len(uplifts_percentiles[0])):
        u_percentiles.append(pd.concat([uplifts_percentiles[j][i] for j in range(5)]).groupby(level=0).mean())
    uplifts_percentiles = u_percentiles
    print('average qini by rank: ', np.mean(qinis))
    print('average auuc by rank: ', np.mean(auucs))
    print('average u at 0.3 by rank: ', np.mean(uks))

    with open(f'{source_dir}/results.txt', 'a') as f:
        f.write('\n========= By Rank =================')
        f.write(f'\nby rank -- qini: {np.mean(qinis)}, auuc: {np.mean(auucs)}, u_0.3: {np.mean(uks)}')
        qinis = [str(round(i, 4)) for i in qinis]
        auucs = [str(round(i, 4)) for i in auucs]
        uks = [str(round(i, 4)) for i in uks]
        f.write(f'\n{" ".join(qinis)}')
        f.write(f'\n{" ".join(auucs)}')
        f.write(f'\n{" ".join(uks)}')

    dfs = pd.concat(uplifts_percentiles)
    dfs = dfs.groupby(level=0).mean()
    print('average percentile by rank: ', dfs)
    dfs.to_csv(f'{source_dir}/uplift_perc_rank.txt', sep='\t')

    plt.figure()
    plt.plot(range(len(rank_names)), qinis)
    plt.xlabel('rank')
    plt.ylabel('QINI')
    plt.xticks(range(len(rank_names)), rank_names, rotation='vertical')
    plt.tight_layout()
    plt.grid(True)
    for i, value in enumerate(qinis):
        plt.annotate(str(value), xy=(i, value), textcoords='offset points', xytext=(0, 10), ha='center')
    plt.savefig(f'{source_dir}/qini_rank.png')

    plt.figure()
    plt.plot(range(len(rank_names)), auucs)
    plt.xlabel('rank')
    plt.ylabel('AUUC')
    plt.xticks(range(len(rank_names)), rank_names, rotation='vertical')
    plt.tight_layout()
    plt.grid(True)
    for i, value in enumerate(auucs):
        plt.annotate(str(value), xy=(i, value), textcoords='offset points', xytext=(0, 10), ha='center')
    plt.savefig(f'{source_dir}/auuc_rank.png')

    plt.figure()
    plt.plot(range(len(rank_names)), uks)
    plt.xlabel('rank')
    plt.ylabel('u_at_k')
    plt.xticks(range(len(rank_names)), rank_names, rotation='vertical')
    plt.tight_layout()
    for i, value in enumerate(uks):
        plt.annotate(str(value), xy=(i, value), textcoords='offset points', xytext=(0, 10), ha='center')
    plt.savefig(f'{source_dir}/uk_rank.png')


metric = 'u_at_k'
treat_name = ''
task_name = 'login'
model_name = 'MTask-mtmt_mmoe_emb_v1_EFIN_l1+0.2l5_diffBi_meanLoss'
data_type = 'mtask'
source = f'predictions/{data_type}/zscore/{model_name}/test/'

print(model_name, metric, treat_name, task_name)

source_dir = os.path.join('results', model_name, data_type, task_name, metric, treat_name)
if not os.path.isdir(source_dir):
    os.makedirs(source_dir)

targets, preds, treats, add_feats = [], [], [], []

u_at_k = qini = auuc = 0
roc = pr = 0
ind_ai = None

add_feat = np.load(source + 'add_features.npz')['feature']
for i in range(5):
    if treat_name != '' or task_name != '':
        if treat_name != '' and task_name != '':
            ...
        elif treat_name != '':
            path = source + f'{model_name}_{i}_{metric}_{treat_name}.npz'
        else:
            path = source + f'{model_name}_{i}_{task_name}_{metric}.npz'
    else:
        path = source + f'{model_name}_{i}_{metric}.npz'
        
    predictions = np.load(path)
    target, pred, treat = predictions['target'], predictions['pred'], predictions['treat']
    
    if treat_name != '':
        ind_control = treat[:, 0] == 0
        if '5AI' in treat_name:
            ind_ai = (treat[:, 0] == 1) & (treat[:, 1] == 0)
            target, pred, treat = target[ind_control | ind_ai], pred[ind_control | ind_ai], treat[:, 0][ind_control | ind_ai]
        else:
            ind_ai = (treat[:, 0] == 1) & (treat[:, 1] == 1)
            target, pred, treat = target[ind_control | ind_ai], pred[ind_control | ind_ai], treat[:, 0][ind_control | ind_ai]
    if task_name != '':
        target = target[:, 0] if task_name == 'login' else target[:, 1]

    targets.append(target)
    preds.append(pred)
    treats.append(treat)
    u_at_k += predictions['u_at_k']

    qini += predictions['QINI']
    auuc += predictions['AUUC']

    r, p = calc_roc_pr(target, pred)
    roc += r
    pr += p

    if ind_ai is not None:
        add_feats.append(add_feat[ind_control | ind_ai])
    else:
        add_feats.append(add_feat)

print('qini: {}, auuc: {}, u at 0.3: {}, roc: {}, pr: {}'.format(qini / 5, auuc / 5, u_at_k / 5, roc / 5, pr / 5))
with open(f'{source_dir}/results.txt', 'w') as f:
    f.write('qini: {}, auuc: {}, u at 0.3: {}, roc: {}, pr: {}'.format(qini / 5, auuc / 5, u_at_k / 5, roc / 5, pr / 5))

analysis_by_logindays(targets, preds, treats, add_feats)
analysis_by_rank(targets, preds, treats, add_feats)
analysis_and_plot_uplift_qini_curves(targets, preds, treats)
