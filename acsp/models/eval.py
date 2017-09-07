# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import codecs
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr

from acsp.com.common import *
from acsp.com.utils import *


def mbias(x, y, axis=None):
    x = np.array(x, dtype=float).flatten()
    y = np.array(y, dtype=float).flatten()
    return np.mean(x-y, axis=axis)


def mvariance(x, axis=None):
    x = np.array(x, dtype=float).flatten()
    return np.var(x, axis=axis)


def mrms(x, y, axis=None):
    x = np.array(x, dtype=float).flatten()
    y = np.array(y, dtype=float).flatten()
    return np.sqrt(np.mean((x - y) ** 2, axis=axis))


def mmse(x, y, axis=None):
    x = np.array(x, dtype=float).flatten()
    y = np.array(y, dtype=float).flatten()
    return np.mean((x - y) ** 2, axis=axis)


def calculate_experiment_1_1():
    """
    [RQ1] How does active sampling perform compared to other sample-based and active-selection methods
    regarding bias and variance in the calculated effectiveness measures?
    """

    trecs = ['TREC-5', 'TREC-6', 'TREC-7', 'TREC-8', 'TREC-9', 'TREC-10', 'TREC-11']
    models = ['mtf', 'mab', 'importance', 'activewr']
    measures = ['ap', 'rp', 'p30', 'dcg']

    for i, trec_name in enumerate(trecs):
        for k, model in enumerate(models):

            # create file
            ret_dir = os.path.join(EXP_DIR, trec_name)
            if not os.path.exists(ret_dir):
                os.makedirs(ret_dir)

            f = codecs.open(os.path.join(ret_dir, '{}.exp1.csv'.format(model)), 'w', encoding='utf-8')
            f_csv = csv.writer(f)
            f_csv.writerow(('trec_name', 'model', 'percentage', 'measure', 'system',
                            'bias', 'variance', 'mse', 'estm', 'actu'))

            for percentage in PERCENTAGES:
                # read data
                list_df = []
                for sample_index in range(1, 31):
                    sample_dir = '{}{}'.format('sample', sample_index)
                    percentage_dir = '{}{}'.format('percentage', int(percentage * 100))
                    ret_dir = os.path.join(RESULT_DIR, trec_name, sample_dir, percentage_dir)
                    file_name = '{}.csv'.format(model)
                    if not os.path.exists(os.path.join(ret_dir, file_name)):
                        print(trec_name, model, sample_index, percentage)
                        continue
                    df = pd.read_csv(os.path.join(ret_dir, file_name))
                    list_df.append(df)

                for j, m in enumerate(measures):
                    actu_m = 'actu_' + m
                    estm_m = 'estm_' + m

                    # calculate statistics
                    for s, sys in enumerate(list_df[0].system):
                        list_estm = [df.loc[df.system == sys].ix[:, estm_m].values[0] for df in list_df]
                        list_actu = [df.loc[df.system == sys].ix[:, actu_m].values[0] for df in list_df]

                        bias_ = mbias(list_estm, list_actu)
                        variance_ = mvariance(list_estm)
                        mse_ = mmse(list_estm, list_actu)
                        estm_ = np.mean(list_estm)
                        actu_ = np.mean(list_actu)

                        f_csv.writerow(
                            (trec_name, model, percentage, m, sys, bias_, variance_, mse_, estm_, actu_))
            f.close()
    return


def calculate_experiment_2_1():
    """
    [RQ2] How fast active sampling estimators approximate the actual evaluation measures
    compared to other sample-based and active-selection methods?
    :return:
    """
    trecs = ['TREC-5', 'TREC-6', 'TREC-7', 'TREC-8', 'TREC-9', 'TREC-10', 'TREC-11']
    models = ['mtf', 'mab', 'importance', 'activewr']
    measures = ['ap', 'rp', 'p30', 'dcg']

    for i, trec_name in enumerate(trecs):
        for k, model in enumerate(models):

            # create file
            ret_dir = os.path.join(EXP_DIR, trec_name)
            if not os.path.exists(ret_dir):
                os.makedirs(ret_dir)

            f = codecs.open(os.path.join(ret_dir, '{}.exp2.csv'.format(model)), 'w', encoding='utf-8')
            f_csv = csv.writer(f)
            f_csv.writerow(('trec_name', 'model', 'percentage', 'measure',
                            'estm_r', 'smpl_r', 'actu_r', 'smpl_doc', 'actu_doc',
                            'bias', 'variance', 'rms', 'rms_var', 'tau', 'tau_var', 'rho'
                            ))

            for percentage in PERCENTAGES:
                # read data
                list_df = []
                for sample_index in range(1, 31):
                    sample_dir = '{}{}'.format('sample', sample_index)
                    percentage_dir = '{}{}'.format('percentage', int(percentage * 100))
                    ret_dir = os.path.join(RESULT_DIR, trec_name, sample_dir, percentage_dir)
                    file_name = '{}.csv'.format(model)
                    if not os.path.exists(os.path.join(ret_dir, file_name)):
                        print(trec_name, model, sample_index, percentage)
                        continue
                    df = pd.read_csv(os.path.join(ret_dir, file_name))
                    list_df.append(df)

                for j, m in enumerate(measures):
                    actu_m = 'actu_' + m
                    estm_m = 'estm_' + m

                    # calculate 'estm_r', 'smpl_r', 'actu_r', 'smpl_doc', 'actu_doc'
                    estm_r_ = np.mean([df.ix[:, 'estm_r'].values[0] for df in list_df])
                    smpl_r_ = np.mean([df.ix[:, 'smpl_r'].values[0] for df in list_df])
                    actu_r_ = np.mean([df.ix[:, 'actu_r'].values[0] for df in list_df])
                    smpl_doc_ = np.mean([df.ix[:, 'smpl_doc'].values[0] for df in list_df])
                    actu_doc_ = np.mean([df.ix[:, 'actu_doc'].values[0] for df in list_df])

                    # calculate bias variance: loop sample first and system second
                    stats = []
                    for s, sys in enumerate(list_df[0].system):
                        list_estm = [df.loc[df.system == sys].ix[:, estm_m].values[0] for df in list_df]
                        list_actu = [df.loc[df.system == sys].ix[:, actu_m].values[0] for df in list_df]

                        bias_ = mbias(list_estm, list_actu)
                        variance_ = mvariance(list_estm)

                        stats.append((bias_, variance_))
                    bias_ = np.mean([b for b, v in stats])
                    variance_ = np.mean([v for b, v in stats])

                    # Calculate mrms, tau, rho: loop system first and sample second
                    stats = []
                    for df in list_df:
                        list_estm = df.ix[:, estm_m].values
                        list_actu = df.ix[:, actu_m].values

                        rms_ = mrms(list_estm, list_actu)
                        tau_, p_value = kendalltau(list_estm, list_actu)
                        rho_, p_value = pearsonr(list_estm, list_actu)

                        stats.append((rms_, tau_, rho_))

                    rms_ = np.mean([r for r, t, h in stats])
                    rms_var = np.var([r for r, t, h in stats])
                    tau_ = np.mean([t for r, t, h in stats])
                    tau_var = np.var([t for r, t, h in stats])
                    rho_ = np.mean([h for r, t, h in stats])

                    f_csv.writerow((trec_name, model, percentage, m, estm_r_, smpl_r_, actu_r_, smpl_doc_, actu_doc_,
                                    bias_, variance_, rms_, rms_var, tau_, tau_var, rho_))
            f.close()
    return


def calculate_experiment_3_1():
    """ [RQ3] Is the test collection generated by active sampling reusable for new runs
    that do not contribute in the construction of the collection?"""

    trecs = ['TREC-5', 'TREC-6', 'TREC-7', 'TREC-8', 'TREC-9', 'TREC-10', 'TREC-11']
    models = ['mtf', 'mab', 'importance', 'activewr', 'activewresap']
    measures = ['ap', 'rp', 'p30', 'dcg']

    for i, trec_name in enumerate(trecs):
        for k, model in enumerate(models):

            # create file
            ret_dir = os.path.join(EXP_DIR, trec_name)
            if not os.path.exists(ret_dir):
                os.makedirs(ret_dir)

            f = codecs.open(os.path.join(ret_dir, '{}.exp3.group.csv'.format(model)), 'w', encoding='utf-8')
            f_csv = csv.writer(f)
            f_csv.writerow(('trec_name', 'type', 'model', 'percentage', 'measure', 'rms', 'tau', 'rho'))

            for percentage in PERCENTAGES:
                # read sample data
                list_df = []
                for split_index in range(0, 42):  # maximum groups number in LIST_GROUPS
                    sample_dir = '{}{}'.format('split_leave-one-group-out_', split_index)
                    percentage_dir = '{}{}'.format('percentage', int(percentage * 100))
                    ret_dir = os.path.join(RESULT_DIR, trec_name, sample_dir, percentage_dir)
                    file_name = '{}.csv'.format(model)
                    if not os.path.exists(os.path.join(ret_dir, file_name)):
                        # print(trec_name, model, split_index, percentage)
                        continue
                    df = pd.read_csv(os.path.join(ret_dir, file_name))
                    list_df.append(df)

                for j, m in enumerate(measures):

                    actu_m = 'actu_' + m
                    estm_m = 'estm_' + m

                    # calculate rms, tau, rho: loop system first and sample second
                    stats = []
                    for df in list_df:
                        list_estm = df.ix[:, estm_m].values
                        list_actu = df.ix[:, actu_m].values

                        rms_ = mrms(list_estm, list_actu)
                        tau_, p_value = kendalltau(list_estm, list_actu)
                        rho_, p_value = pearsonr(list_estm, list_actu)

                        stats.append((rms_, tau_, rho_))

                    rms_ = np.mean([r for r, t, h in stats if r is not np.nan])
                    tau_ = np.mean([t for r, t, h in stats if t is not np.nan])
                    rho_ = np.mean([h for r, t, h in stats if h is not np.nan])

                    f_csv.writerow((trec_name, 'null', model, percentage, m, rms_, tau_, rho_))
            f.close()
    return


if __name__ == '__main__':
    # calculate_experiment_1_1()
    # calculate_experiment_2_1()
    # calculate_experiment_3_1()

    pass