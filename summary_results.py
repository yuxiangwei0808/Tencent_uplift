from sklift.metrics import uplift_auc_score, qini_auc_score, uplift_by_percentile
from metrics import uplift_at_k, weighted_average_uplift
from sklearn.metrics import *
import numpy as np
import torch

metrics = ['QINI' 'u_at_k']
model_name = 'MTask-mtmt_mmoe_emb_v1_EFIN_l1+0.2l5_diffBi_meanLoss'
dir = f'predictions/mtask/zscore/{model_name}/test/'
source = dir + f'{model_name}_'

for metric in metrics:
    print(f'============ {metric} =====================')
    for task in ['login']:
        print(f'+++++++++ {task} ++++++++')
        qini =[]
        auuc = []
        roc = []
        pr = []
        u_at_k  = []

        for i in range(5):
            x = np.load(source + str(i) + f'_{task}_{metric}.npz')
            true_labels = x['target']
            predictions = x['pred']
            is_treatment = x['treat']

            true_labels = true_labels[:, 0] if task == 'login' else true_labels[:, 1]

            qini.append(round(qini_auc_score(true_labels, predictions, is_treatment), 3))
            auuc.append(round(uplift_auc_score(true_labels, predictions, is_treatment), 3))
            u_at_k.append(round(uplift_at_k(true_labels, predictions, is_treatment, strategy='overall', k=0.3), 3))

            # print(uplift_by_percentile(true_labels, predictions, is_treatment))

            prob = np.array(torch.sigmoid(torch.tensor(predictions)))
            roc.append(roc_auc_score(true_labels, prob))
            pr.append(average_precision_score(true_labels, prob))

        print(f'qini: {qini}, auuc: {auuc}, u_at_0.3: {u_at_k}')
        print(f'qini: {sum(qini) / len(qini)}, auuc: {sum(auuc) / len(auuc)}, u_at_0.3: {sum(u_at_k) / len(u_at_k)}')
        # with open(dir + 'summary.txt', 'w') as f:
        #     f.write(f'\n Metric -- {metric}, qini: {sum(qini) / len(qini)}, auuc: {sum(auuc) / len(auuc)}, roc:{sum(roc) / len(roc)}, pr:{sum(pr) / len(pr)}, u_at_0.3: {sum(u_at_k) / len(u_at_k)}')