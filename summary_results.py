from sklift.metrics import uplift_auc_score, qini_auc_score, uplift_by_percentile
from metrics import uplift_at_k, weighted_average_uplift
from sklearn.metrics import *
import numpy as np
import torch

metrics = ['QINI', 'u_at_k']

def summary():
    metrics = ['QINI', 'u_at_k']
    model_name = 'mtmt_res_emb_v0_4_0_EFIN_l1+0.2l5_noEnhance'
    dir = f'predictions/random/zscore/{model_name}/test/'
    print(model_name)

    for metric in metrics:
        source = dir + f'{model_name}_'
        qini =[]
        auuc = []
        u_at_k  = []

        for i in range(5):
            x = np.load(source + str(i) + f'_{metric}.npz')
            true_labels = x['target']
            predictions = x['pred']
            is_treatment = x['treat']

            qini.append(x['QINI'])
            auuc.append(x['AUUC'])
            u_at_k.append(x['u_at_k'])

        print(f'qini: {sum(qini) / len(qini)}, auuc: {sum(auuc) / len(auuc)}, u_at_0.3: {sum(u_at_k) / len(u_at_k)}')


def multi_task():
    model_name = 'MTask-mtmt_mmoe_emb_v2_EFIN_l1+0.2l5_diffBi_taskWiseTauL5_dwa'
    dir = f'predictions/mtask/zscore/{model_name}/test/'
    source = dir + f'{model_name}_'
    for metric in metrics:
        print(f'============ {metric} =====================')
        for task in ['login', 'diff']:
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
                predictions = predictions[:, 0] if task == 'login' else predictions[:, 1]

                qini.append(round(qini_auc_score(true_labels, predictions, is_treatment), 3))
                auuc.append(round(uplift_auc_score(true_labels, predictions, is_treatment), 3))
                u_at_k.append(round(uplift_at_k(true_labels, predictions, is_treatment, strategy='overall', k=0.3), 3))

            print(f'qini: {qini}, auuc: {auuc}, u_at_0.3: {u_at_k}')
            print(f'qini: {sum(qini) / len(qini)}, auuc: {sum(auuc) / len(auuc)}, u_at_0.3: {sum(u_at_k) / len(u_at_k)}')


def multi_treat():
    model_name = 'm3tn_criteo'
    dir = f'predictions/warmtype/zscore/{model_name}/train/'
    source = dir + f'{model_name}_'
    for metric in metrics:
        print(f'============ {metric} =====================')
        for treat_name in ['5AI', '9AI']:
            print(f'+++++++++ {treat_name} ++++++++')
            qini =[]
            auuc = []
            u_at_k  = []

            for i in range(5):
                # x = np.load(source + f'{i}_{metric}_{treat_name}.npz')
                x = np.load(source + f'{i}_{metric}.npz')
                target = x['target']
                pred = x['pred']
                treat = x['treat']

                ind_control = treat[:, 0] == 0
                if '5AI' in treat_name:
                    ind_ai = (treat[:, 0] == 1) & (treat[:, 1] == 0)
                else:
                    ind_ai = (treat[:, 0] == 1) & (treat[:, 1] == 1)

                target, pred, treat = target[ind_control | ind_ai], pred[ind_control | ind_ai], treat[:, 0][ind_control | ind_ai]

                qini.append(round(qini_auc_score(target, pred, treat), 3))
                auuc.append(round(uplift_auc_score(target, pred, treat), 3))
                u_at_k.append(round(uplift_at_k(target, pred, treat, strategy='overall', k=0.3), 3))

            print(f'qini: {qini}, auuc: {auuc}, u_at_0.3: {u_at_k}')
            print(f'qini: {sum(qini) / len(qini)}, auuc: {sum(auuc) / len(auuc)}, u_at_0.3: {sum(u_at_k) / len(u_at_k)}')


def mtmt():
    model_name = 'MTask_mtmt_mmoe_emb_v2_EFIN'
    dir = f'predictions/warmtype/zscore/{model_name}/test/'
    source = dir + f'{model_name}_'
    for metric in metrics:
        print(f'============ {metric} =====================')
        for task_name in ['login', 'diff']:
            print(f'------------- {task_name} -------------------')
            for treat_name in ['5AI', '9AI']:
                print(f'+++++++++++++ {treat_name} ++++++++++++++++++')
                qini =[]
                auuc = []
                u_at_k  = []

                for i in range(5):
                    x = np.load(source + f'{i}_{task_name}_{metric}_{treat_name}.npz')
                    target = x['target']
                    pred = x['pred']
                    treat = x['treat']

                    ind_control = treat[:, 0] == 0
                    if '5AI' in treat_name:
                        ind_ai = (treat[:, 0] == 1) & (treat[:, 1] == 0)
                    else:
                        ind_ai = (treat[:, 0] == 1) & (treat[:, 1] == 1)

                    target, pred, treat = target[ind_control | ind_ai], pred[ind_control | ind_ai], treat[:, 0][ind_control | ind_ai]

                    target = target[:, 0] if task_name == 'login' else target[:, 1]
                    pred = pred[:, 0] if task_name == 'login' else pred[:, 1]

                    qini.append(round(qini_auc_score(target, pred, treat), 3))
                    auuc.append(round(uplift_auc_score(target, pred, treat), 3))
                    u_at_k.append(round(uplift_at_k(target, pred, treat, strategy='overall', k=0.3), 3))

                print(f'qini: {qini}, auuc: {auuc}, u_at_0.3: {u_at_k}')
                print(f'qini: {sum(qini) / len(qini)}, auuc: {sum(auuc) / len(auuc)}, u_at_0.3: {sum(u_at_k) / len(u_at_k)}')


mtmt()