import torch
import numpy as np
import pandas as pd


def metrics_mt(metric, y_true, uplift, treat, m_treat=False, reduce=None):
    # multi-treatment version
    # `treat` should be N*T, where the first element denotes treatment/control

    if not m_treat:
        return metric(y_true, uplift, treat)

    ind_control = treat[:, 0] == 0
    ind_5ai = (treat[:, 0] == 1) & (treat[:, 1] == 0)
    ind_9ai = (treat[:, 0] == 1) & (treat[:, 1] == 1)

    score_5ai = metric(y_true[ind_control | ind_5ai], uplift[ind_control | ind_5ai], treat[:, 0][ind_control | ind_5ai])
    score_9ai = metric(y_true[ind_control | ind_9ai], uplift[ind_control | ind_9ai], treat[:, 0][ind_control | ind_9ai])

    if reduce == 'max':
        return max(score_5ai, score_9ai)
    elif reduce == 'mean':
        return (score_5ai + score_9ai) * 0.5
    else:
        return score_5ai, score_9ai


def uplift_at_k(y_true, uplift, treatment, strategy='overall', k=0.3):
    n_samples = len(y_true)
    order = np.argsort(uplift, kind='mergesort')[::-1]

    if strategy == 'overall':
        n_size = int(n_samples * k)

        # ToDo: _checker_ there are observations among two groups among first k
        score_ctrl = y_true[order][:n_size][treatment[order][:n_size] == 0].mean()
        score_trmnt = y_true[order][:n_size][treatment[order][:n_size] == 1].mean()
        if np.isnan(score_ctrl):
            score_ctrl = 0
        if np.isnan(score_trmnt):
            score_trmnt = 0

    else:  # strategy == 'by_group':
        n_ctrl = int((treatment == 0).sum() * k)
        n_trmnt = int((treatment == 1).sum() * k)

        score_ctrl = y_true[order][treatment[order] == 0][:n_ctrl].mean()
        score_trmnt = y_true[order][treatment[order] == 1][:n_trmnt].mean()

        if np.isnan(score_ctrl):
            score_ctrl = 0
        if np.isnan(score_trmnt):
            score_trmnt = 0

    return score_trmnt - score_ctrl


def response_rate_by_percentile(y_true, uplift, treatment, group, strategy='overall', bins=10):
    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)
    order = np.argsort(uplift, kind='mergesort')[::-1]

    trmnt_flag = 1 if group == 'treatment' else 0

    if strategy == 'overall':
        y_true_bin = np.array_split(y_true[order], bins)
        trmnt_bin = np.array_split(treatment[order], bins)

        group_size = np.array([len(y[trmnt == trmnt_flag]) for y, trmnt in zip(y_true_bin, trmnt_bin)])
        response_rate = np.array([np.mean(y[trmnt == trmnt_flag]) for y, trmnt in zip(y_true_bin, trmnt_bin)])

    else:  # strategy == 'by_group'
        y_bin = np.array_split(y_true[order][treatment[order] == trmnt_flag], bins)

        group_size = np.array([len(y) for y in y_bin])
        response_rate = np.array([np.mean(y) for y in y_bin])

    response_rate[np.isnan(response_rate)] = 0
    _group_size = group_size.copy()
    _group_size[_group_size == 0] = 1
    variance = np.multiply(response_rate, np.divide((1 - response_rate), _group_size))

    return response_rate, variance, group_size


def weighted_average_uplift(y_true, uplift, treatment, strategy='overall', bins=10):
    response_rate_trmnt, variance_trmnt, n_trmnt = response_rate_by_percentile(
        y_true, uplift, treatment, group='treatment', strategy=strategy, bins=bins)

    response_rate_ctrl, variance_ctrl, n_ctrl = response_rate_by_percentile(
        y_true, uplift, treatment, group='control', strategy=strategy, bins=bins)

    uplift_scores = response_rate_trmnt - response_rate_ctrl

    weighted_avg_uplift = np.dot(n_trmnt, uplift_scores) / np.sum(n_trmnt)

    return weighted_avg_uplift


def uplift_by_percentile(y_true, uplift, treatment, strategy='overall',
                         bins=10, std=False, total=False, string_percentiles=True):
    response_rate_trmnt, variance_trmnt, n_trmnt = response_rate_by_percentile(
        y_true, uplift, treatment, group='treatment', strategy=strategy, bins=bins)

    response_rate_ctrl, variance_ctrl, n_ctrl = response_rate_by_percentile(
        y_true, uplift, treatment, group='control', strategy=strategy, bins=bins)

    uplift_scores = response_rate_trmnt - response_rate_ctrl
    uplift_variance = variance_trmnt + variance_ctrl

    percentiles = [round(p * 100 / bins) for p in range(1, bins + 1)]

    if string_percentiles:
        percentiles = [f"0-{percentiles[0]}"] + \
            [f"{percentiles[i]}-{percentiles[i + 1]}" for i in range(len(percentiles) - 1)]

    df = pd.DataFrame({
        'percentile': percentiles,
        'n_treatment': n_trmnt,
        'n_control': n_ctrl,
        'response_rate_treatment': response_rate_trmnt,
        'response_rate_control': response_rate_ctrl,
        'uplift': uplift_scores
    })

    if total:
        response_rate_trmnt_total, variance_trmnt_total, n_trmnt_total = response_rate_by_percentile(
            y_true, uplift, treatment, strategy=strategy, group='treatment', bins=1)

        response_rate_ctrl_total, variance_ctrl_total, n_ctrl_total = response_rate_by_percentile(
            y_true, uplift, treatment, strategy=strategy, group='control', bins=1)

        df.loc[-1, :] = ['total', n_trmnt_total, n_ctrl_total, response_rate_trmnt_total,
                         response_rate_ctrl_total, response_rate_trmnt_total - response_rate_ctrl_total]

    if std:
        std_treatment = np.sqrt(variance_trmnt)
        std_control = np.sqrt(variance_ctrl)
        std_uplift = np.sqrt(uplift_variance)

        if total:
            std_treatment = np.append(std_treatment, np.sum(std_treatment))
            std_control = np.append(std_control, np.sum(std_control))
            std_uplift = np.append(std_uplift, np.sum(std_uplift))

        df.loc[:, 'std_treatment'] = std_treatment
        df.loc[:, 'std_control'] = std_control
        df.loc[:, 'std_uplift'] = std_uplift

    df = df \
        .set_index('percentile', drop=True, inplace=False) \
        .astype({'n_treatment': 'int32', 'n_control': 'int32'})

    return df