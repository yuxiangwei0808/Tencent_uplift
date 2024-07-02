import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from metrics import uplift_at_k, uplift_by_percentile
from sklift.metrics import qini_auc_score, uplift_auc_score
from sklift.viz import plot_qini_curve, plot_uplift_by_percentile, plot_uplift_curve


def sigmoid(z):
    return 1/(1 + np.exp(-z))


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
    uplift_perc, qini, auuc, u_at_k = analysis(targets, predictions, treats)
    print(uplift_perc)
    print('qini: {}, auuc: {}, u at 0.3: {}'.format(qini, auuc, u_at_k))
    
    plt.figure()
    sns.displot(predictions, bins=100, kde=True)
    # plt.xlim(-0.1, 0.3)
    plt.grid(True)
    plt.savefig('u_trend.png', bbox_inches='tight') 
    
    up_per = plot_uplift_by_percentile(targets, predictions, treats, kind='bar')
    plt.savefig('uplift_by_percentile.png', bbox_inches='tight')
    
    qini_disp = plot_qini_curve(targets, predictions, treats)
    plt.savefig('qini.png', bbox_inches='tight')
    
    auuc_disp = plot_uplift_curve(targets, predictions, treats)
    plt.savefig('uplift.png', bbox_inches='tight')


def analysis(targets, preds, treats):
    preds_bin = sigmoid(preds)
    preds_bin[preds_bin >= 0.5] = 1
    preds_bin[preds_bin < 0.5] = 0
    
    # print('==================== Classification Report ====================')
    # print(classification_report(targets, preds_bin))

    # print('==================== Confusion Matrix ====================')
    # print(confusion_matrix(targets, preds_bin))

    uplifts = uplift_by_percentile(targets, preds, treats)
    qini = qini_auc_score(targets, preds, treats)
    auuc = uplift_auc_score(targets, preds, treats)
    u_at_k = uplift_at_k(targets, preds, treats, strategy='overall', k=0.3)

    # plot_and_save_roc_auc(preds, targets)
    # plot_and_save_pr_auc(preds, targets)
    # plot_and_save_precision_recall_vs_threshold(preds, targets)
    
    return uplifts, qini, auuc, u_at_k


def analysis_by_logindays(targets, preds, treats, add_feats):
    # analyze the results by each login days (add_feats[:, 1])
    uplifts_percentiles, qinis, auucs, uks = [], [], [], []
    
    if 'full' in source:
        ranger = np.arange(15)
    elif 'highactive' in source:
        ranger = np.arange(10, 15)
    elif 'midactive' in source:
        ranger = np.arange(5, 10)
    elif 'lowactive' in source:
        ranger = np.arange(1, 5)
    else:
        ranger = np.arange(1)
    
    for day in ranger:
        idx = add_feats == day
        uplifts, qini, auuc, u_at_k = analysis(targets[idx], preds[idx], treats[idx])
        uplifts_percentiles.append(uplifts)
        qinis.append(qini)
        auucs.append(auuc)
        uks.append(u_at_k)
        
        percentiles = list(uplifts['uplift'].keys())
            
    print('average qini: ', np.mean(qinis))
    print('average auuc: ', np.mean(auucs))
    print('average u at 0.3: ', np.mean(uks))
    
    dfs = pd.concat(uplifts_percentiles)
    print('average percentile: ', dfs.groupby(level=0).mean())
    
    plt.figure()
    plt.plot(ranger, qinis)
    plt.grid(True)
    plt.xlabel('login days')
    plt.ylabel('QINI')
    plt.savefig('qini_login.png')
    
    plt.figure()
    plt.plot(ranger, auucs)
    plt.grid(True)
    plt.xlabel('login days')
    plt.ylabel('AUUC')
    plt.savefig('auuc_login.png')
    
    fig, ax = plt.subplots()
    for p in percentiles:
        ufs = [x.loc[p]['uplift'] for x in uplifts_percentiles]
        ax.plot(ranger, ufs, label=f'percentile={p}')
    ax.legend(fontsize='5')
    ax.grid(True)
    ax.set_xlabel('login days')
    ax.set_ylabel('uplifts')
    plt.savefig('uplift_by_percentile_login.png')


metric = 'QINI'
model_name = 'mtmt_res_emb_v0_4_0'
source = f'predictions/full/zscore/{model_name}/test/'

targets, preds, treats, add_feats = [], [], [], []

        
for i in range(5):
    predictions = np.load(source + f'{model_name}_{i}_{metric}.npz')
    target, pred, treat = predictions['target'], predictions['pred'], predictions['treat']
    add_feat = np.load(source + 'add_features.npz')['feature'][:, 1]
    targets.append(target)
    preds.append(pred)
    treats.append(treat)
    add_feats.append(add_feat)

targets, preds, treats, add_feats = np.concatenate(targets, 0), np.concatenate(preds, 0), np.concatenate(treats, 0), np.concatenate(add_feats, 0)

analysis_and_plot_uplift_qini_curves(targets, preds, treats)
analysis_by_logindays(targets, preds, treats, add_feats)