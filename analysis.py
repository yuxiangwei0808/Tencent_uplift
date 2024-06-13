import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt
import pandas as pd

from metrics import uplift_at_k, uplift_by_percentile
from sklift.metrics import qini_auc_score, uplift_auc_score


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

    # plot_and_save_roc_auc(preds, targets)
    # plot_and_save_pr_auc(preds, targets)
    # plot_and_save_precision_recall_vs_threshold(preds, targets)
    
    return uplifts, qini, auuc


def analysis_by_logindays(targets, preds, treats, add_feats):
    # analyze the results by each login days (add_feats[:, 1])
    uplifts_percentiles, qinis, auucs = [], [], []    
    for day in range(15):  # 0-14
        idx = add_feats == day
        uplifts, qini, auuc = analysis(targets[idx], preds[idx], treats[idx])
        uplifts_percentiles.append(uplifts)
        qinis.append(qini)
        auucs.append(auuc)
        
        if day == 0:
            percentiles = list(uplifts['uplift'].keys())
            
    print('average qini: ', np.mean(qinis))
    print('average auuc: ', np.mean(auucs))
    
    dfs = pd.concat(uplifts_percentiles)
    print('average percentile: ', dfs.groupby(level=0).mean())
    
    plt.figure()
    plt.plot(np.arange(15), qinis)
    plt.grid(True)
    plt.xlabel('login days')
    plt.ylabel('QINI')
    plt.savefig('qini.png')
    
    plt.figure()
    plt.plot(np.arange(15), auucs)
    plt.grid(True)
    plt.xlabel('login days')
    plt.ylabel('AUUC')
    plt.savefig('auuc.png')
    
    fig, ax = plt.subplots()
    for p in percentiles:
        ufs = [x.loc[p]['uplift'] for x in uplifts_percentiles]
        ax.plot(np.arange(15), ufs, label=f'percentile={p}')
    ax.legend(fontsize='5')
    ax.grid(True)
    ax.set_xlabel('login days')
    ax.set_ylabel('uplifts')
    plt.savefig('uplift_by_percentile.png')


metric = 'ROC-AUC'
source = 'predictions/lowactive/minmax/efin/test/efin_96_96_0.001_'

targets, preds, treats, add_feats = [], [], [], []


for i in range(5):
    predictions = np.load(source + f'{i}_{metric}.npz')
    target, pred, treat, add_feat = predictions['target'], predictions['pred'], predictions['treat'], predictions['feature'][:, 1]
    targets.append(target)
    preds.append(pred)
    treats.append(treat)
    add_feats.append(add_feat)

targets, preds, treats, add_feats = np.concatenate(targets, 0), np.concatenate(preds, 0), np.concatenate(treats, 0), np.concatenate(add_feats, 0)

analysis_by_logindays(targets, preds, treats, add_feats)