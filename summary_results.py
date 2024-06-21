from sklift.metrics import uplift_auc_score, qini_auc_score
from metrics import uplift_at_k, weighted_average_uplift
from sklearn.metrics import *
import numpy as np
import torch

metric = 'ROC-AUC'
metrics = ['u_at_k']

for metric in metrics:
    model_name = 'mtmt_res_emb_v0_criteo'
    dir = f'predictions/full/zscore/{model_name}/train/'
    source = dir + f'{model_name}_'

    qini =[]
    auuc = []
    roc = []
    pr = []
    u_at_k  = []

    for i in range(5):
        x = np.load(source + str(i) + f'_{metric}.npz')
        true_labels = x['target']
        predictions = x['pred']
        is_treatment = x['treat']

        qini.append( qini_auc_score(true_labels, predictions, is_treatment))
        auuc.append(uplift_auc_score(true_labels, predictions, is_treatment))

        prob = np.array(torch.sigmoid(torch.tensor(predictions)))
        roc.append(roc_auc_score(true_labels, prob))
        pr.append(average_precision_score(true_labels, prob))
        u_at_k.append(uplift_at_k(true_labels, predictions, is_treatment, strategy='overall', k=0.3))



    # print(f'qini: {qini}, auuc: {auuc}, roc: {roc}, pr: {pr}, u_at_0.3: {u_at_k}')
    print(f'qini: {sum(qini) / len(qini)}, auuc: {sum(auuc) / len(auuc)}, roc:{sum(roc) / len(roc)}, pr:{sum(pr) / len(pr)}, u_at_0.3: {sum(u_at_k) / len(u_at_k)}')
    with open(dir + 'summary.txt', 'a') as f:
        f.write(f'\n Metric -- {metric}, qini: {sum(qini) / len(qini)}, auuc: {sum(auuc) / len(auuc)}, roc:{sum(roc) / len(roc)}, pr:{sum(pr) / len(pr)}, u_at_0.3: {sum(u_at_k) / len(u_at_k)}')