# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, ttest_ind
from acsp.com.common import *
from acsp.com.utils import *

from acsp.pre.preprocess import Preprocess
from acsp.models import basemodel

import getpass
if 'danli' == getpass.getuser():
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-paper')   # https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
    import matplotlib.lines as mlines


def tbl_statistics_test_collection():
    """Statistics of the test collections."""

    print('{} & {} & {} & {}'.format('rel doc', 'total doc', 'avg rel per topic', 'avg total per topic'))

    for key, value in DICT_TREC_TOPIC.items():
        mlist = []
        for topic_id in value:
            p = Preprocess(key, topic_id, None)
            rel, total = p.stats_test_collection()
            mlist.append((rel, total))

        print('{} & {} & {} & {} & {}'.format(key,
              np.sum([r for r, t in mlist]),
              np.sum([t for r, t in mlist]),
              np.mean([r for r, t in mlist]),
              np.mean([t for r, t in mlist])))

    return


def plot_sampling_distribution():
    # dist_a = basemodel.BoltzmannDist().test_prob(100, 1)
    # dist_a1 = basemodel.BoltzmannDist().test_prob(100, 2)
    # dist_a2 = basemodel.BoltzmannDist().test_prob(100, 0.7)
    # dist_a3 = basemodel.BoltzmannDist().test_prob(100, 0.6)
    # dist_a4 = basemodel.BoltzmannDist().test_prob(100, 0.3)
    # dist_a5 = basemodel.BoltzmannDist().test_prob(100, 0.1)
    # dist_a6 = basemodel.BoltzmannDist().test_prob(100, 0.01)
    dist_b = basemodel.APDist().predict_doc_prob(100)
    # plt.plot(dist_a, label=r'Boltzmann-based, $\tau$=1')
    # plt.plot(dist_a1, label=r'Boltzmann-based, $\tau$=2')
    # plt.plot(dist_a2, label=r'Boltzmann-based, $\tau$=0.7')
    # plt.plot(dist_a3, label=r'Boltzmann-based, $\tau$=0.6')
    # plt.plot(dist_a4, label=r'Boltzmann-based, $\tau$=0.2')
    # plt.plot(dist_a5, label=r'Boltzmann-based, $\tau$=0.1')
    # plt.plot(dist_a6, label=r'Boltzmann-based, $\tau$=0.01')
    plt.plot(dist_b, label='AP-prior')
    # plt.ylim([0, 0.2])
    plt.xlabel('Document rank')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid()
    # plt.title('Boltzmann-based distribution V.S. Ap-prior distribution')
    plt.show()

    return


def plot_experiment_1_1():
    """Plot scatters for different models on one sample."""

    trec = 'TREC-5'
    percentage = 'percentage10'

    measures = ['ap', 'rp', 'p30']
    subtitles = ['MAP', 'RP', 'P@30']

    models = ['mtf', 'mab', 'importance', 'activewresap']
    labels = ['MTF', 'MAB', 'Stratif', 'Active']
    colors = ['#0088A8', '#B8860B', '#483D8B', '#DC143C']
    markers = ['^', 'v', 'x', 'o']
    f, axarr = plt.subplots(1, len(measures), sharey=True)

    for k, model in enumerate(models):
        # read one sample
        sample_index = '2'
        sample_dir = '{}{}'.format('sample', sample_index)
        ret_dir = os.path.join(RESULT_DIR, trec, sample_dir, percentage)
        file_name = '{}.csv'.format(model)
        df = pd.read_csv(os.path.join(ret_dir, file_name))

        # plot scatters
        for i, m in enumerate(measures):
            actu_m = 'actu_' + m
            estm_m = 'estm_' + m

            # # pearson rho
            # rho_, p_value = pearsonr(df[estm_m].values, df[actu_m].values)
            # axarr[i].text(0.02+0.02, 0.3, r'$Pearson \ \rho$', fontsize=5)
            # axarr[i].text(0.02, 0.3-(k+1)*0.012, '{}'.format(labels[k]), fontsize=5, family='serif')
            # axarr[i].text(0.02+0.05, 0.3-(k+1)*0.012, ': {:<.4f}'.format(rho_), fontsize=5, family='serif')

            # scatters
            axarr[i].scatter(df[actu_m].values, df[estm_m].values, marker=markers[k], s=16, color=colors[k],
                             alpha=0.8, label=labels[k])
            axarr[i].plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=0.5)

            axarr[i].set_ylim(0, 0.5)
            axarr[i].set_xlim(0, 0.5)

            axarr[i].xaxis.set_tick_params(pad=1, labelsize=10)
            axarr[i].yaxis.set_tick_params(pad=1, labelsize=10)

            axarr[i].set_title(subtitles[i], fontsize=12)

            # grid
            axarr[i].grid(b=True, which='major', color='gray', linestyle='-', alpha=0.25, zorder=1, lw=0.5)

            first_in_row = (i == 0)
            middle_in_row = (i == int(len(measures) / 2))

            if first_in_row:
                axarr[i].set_ylabel('Estimated value', fontsize=12)
            if middle_in_row:
                axarr[i].set_xlabel('Actual value', labelpad=8, fontsize=12)

    plt.legend()
    plt.show()
    return


def plot_experiment_1_2():
    """Plot mse/bias/variance for different models over all samples."""

    trec_name = 'TREC-5'
    percentage = 0.1

    measures = ['ap', 'rp', 'p30']
    subtitles = ['MAP', 'RP', 'P@30']

    stats = ['mse', 'bias', 'variance']
    ylabels = ['MSE', 'Bias', 'Variance']

    models = ['mtf', 'mab', 'importance', 'activewresap']
    labels = ['MTF', 'MAB', 'Stratif', 'Active']
    colors = ['#0088A8', '#B8860B', '#483D8B', '#DC143C']
    markers = ['^', 'v', '*', '.']

    f, axarr = plt.subplots(len(stats), len(measures))

    for k, model in enumerate(models):
        for j, measures_ in enumerate(measures):

            # read data
            df = pd.read_csv(os.path.join(os.path.join(EXP_DIR, trec_name), '{}.exp1.csv'.format(model)))
            df = df.loc[(df['percentage'] == percentage) & (df['measure'] == measures_)]
            systems = df['system'].unique()
            # print(systems.values)

            # extract statistics
            array_stats = np.full((len(systems), len(stats)), fill_value=0, dtype=float)
            for s, sys in enumerate(systems):
                for i, stats_ in enumerate(stats):
                    array_stats[s][i] = df.loc[df['system'] == sys].ix[:, stats_].values[0]

            # plot bias/variance/mse
            for i, stats_ in enumerate(stats):
                if not (('mtf' == model ) and ('variance' == stats_)):

                    axarr[i, j].plot(range(len(systems)), array_stats[:, i].flatten(), label=model, linestyle='--',
                                     linewidth=0.5, color=colors[k], marker=markers[k], markersize=3.5)

                    # tick fontsize & spacing
                    axarr[i, j].xaxis.set_tick_params(pad=1, labelsize=6)
                    axarr[i, j].yaxis.set_tick_params(pad=1, labelsize=6)

                    # grid
                    axarr[i, j].grid(b=True, which='major', color='gray', linestyle='-',
                                     alpha=0.25, zorder=1, lw=0.5)

                    first_row = (i == 0)
                    last_row = (i == (len(stats) - 1))
                    first_in_row = (j == 0)
                    middle_in_row = (j == int(len(measures)/2))

                    if first_row:
                        axarr[i, j].set_title(subtitles[j], fontsize=10)
                    if last_row & middle_in_row:
                        axarr[i, j].set_xlabel('System run', labelpad=8, fontsize=10)
                    if first_in_row:
                        axarr[i, j].set_ylabel(ylabels[i], labelpad=8, fontsize=10)

    # f.suptitle(r"Bias & variance", fontsize=11)

    patch0 = mlines.Line2D([], [], color=colors[0], marker=markers[0], markersize=4, label=labels[0])
    patch1 = mlines.Line2D([], [], color=colors[1], marker=markers[1], markersize=4, label=labels[1])
    patch2 = mlines.Line2D([], [], color=colors[2], marker=markers[2], markersize=4, label=labels[2])
    patch3 = mlines.Line2D([], [], color=colors[3], marker=markers[3], markersize=4, label=labels[3])
    plt.legend(handles=[patch0, patch1, patch2, patch3])

    plt.show()
    return


def hypothesis_test_experiment_1_2():
    """Test whether the mes/bias/variance of active sampling are significantly different from mtf/mab/importance. """

    trec_name = 'TREC-5'
    percentage = 0.1
    stats = 'mse'
    measures = ['ap', 'rp', 'p30']

    for j, measures_ in enumerate(measures):
        # read avg mse over all samples
        df = pd.read_csv(os.path.join(os.path.join(EXP_DIR, trec_name), '{}.exp1.csv'.format('mtf')))
        df = df.loc[(df['percentage'] == percentage) & (df['measure'] == measures_)]
        systems = df['system'].unique()

        mtf_stats = np.array([df.loc[df['system'] == sys].ix[:, stats].values[0] for s, sys in enumerate(systems)])

        df = pd.read_csv(os.path.join(os.path.join(EXP_DIR, trec_name), '{}.exp1.csv'.format('mab')))
        df = df.loc[(df['percentage'] == percentage) & (df['measure'] == measures_)]
        mab_stats = np.array([df.loc[df['system'] == sys].ix[:, stats].values[0] for s, sys in enumerate(systems)])

        df = pd.read_csv(os.path.join(os.path.join(EXP_DIR, trec_name), '{}.exp1.csv'.format('importance')))
        df = df.loc[(df['percentage'] == percentage) & (df['measure'] == measures_)]
        importance_stats = np.array([df.loc[df['system'] == sys].ix[:, stats].values[0] for s, sys in enumerate(systems)])

        df = pd.read_csv(os.path.join(os.path.join(EXP_DIR, trec_name), '{}.exp1.csv'.format('activewresap')))
        df = df.loc[(df['percentage'] == percentage) & (df['measure'] == measures_)]
        activewresap_stats = np.array([df.loc[df['system'] == sys].ix[:, stats].values[0] for s, sys in enumerate(systems)])

        # t test
        t, p = ttest_ind(mtf_stats, activewresap_stats, equal_var=False)
        print('{} mtf {:.4f} {:.4f}'.format(measures_, t, p))
        t, p = ttest_ind(mab_stats, activewresap_stats, equal_var=False)
        print('{} mab {:.4f} {:.4f}'.format(measures_, t, p))
        t, p = ttest_ind(importance_stats, activewresap_stats, equal_var=False)
        print('{} imp {:.4f} {:.4f}'.format(measures_, t, p))

    return


def plot_experiment_2_1():
    """Plot rms/tau for different model over all samples."""

    trecs = ['TREC-5', 'TREC-6', 'TREC-7', 'TREC-8', 'TREC-9', 'TREC-10',  'TREC-11']

    measures = ['ap', 'rp', 'p30']
    subtitles = ['MAP', 'RP', 'P@30']

    models = ['mtf', 'mab', 'importance', 'activewresap']
    labels = ['MTF', 'MAB', 'Stratif', 'Active']
    colors = ['#0088A8', '#B8860B', '#483D8B', '#DC143C']
    markers = ['^', 'v', '*', '.']

    f, axarr = plt.subplots(len(trecs), len(measures))

    for i, trec_name in enumerate(trecs):
        for j, measure in enumerate(measures):
            ax_twin = axarr[i, j].twinx()
            for k, model in enumerate(models):
                df = pd.read_csv(os.path.join(os.path.join(EXP_DIR, trec_name), '{}.exp2.csv'.format(model)))
                perc = df.loc[df['measure'] == measure].ix[:, 'percentage']

                # left y-axis
                rms = df.loc[df['measure'] == measure].ix[:, 'rms']
                rms_var = df.loc[df['measure'] == measure].ix[:, 'rms_var']

                axarr[i, j].plot(perc, rms, linestyle='-', label=model,
                                 linewidth=0.4, color=colors[k], marker=markers[k], markersize=2)

                # axarr[i, j].fill_between(perc, rms - rms_var, rms + rms_var, color=colors[k], alpha=0.8)

                # right y-axis
                tau = df.loc[df['measure'] == measure].ix[:, 'tau']
                tau_var = df.loc[df['measure'] == measure].ix[:, 'tau_var']
                ax_twin.plot(perc, tau, linestyle='--', label=model,
                             linewidth=0.4, color=colors[k], marker=markers[k], markersize=2)

                # ax_twin.fill_between(perc, tau - tau_var, tau + tau_var, color=colors[k], alpha=0.8)

            # tick fontsize & spacing
            axarr[i, j].xaxis.set_tick_params(pad=1, labelsize=4)
            axarr[i, j].yaxis.set_tick_params(pad=1, labelsize=4)
            ax_twin.yaxis.set_tick_params(pad=1, labelsize=4)

            # grid
            axarr[i, j].grid(b=True, which='major', color='gray', linestyle='-',
                             alpha=0.25, zorder=1, lw=0.2)

            first_row = (i == 0)
            last_row = (i == len(trecs)-1)
            first_in_row = (j == 0)
            middle_in_row = (j == int(len(measures)/2))

            if first_row:
                axarr[i, j].set_title(subtitles[j], fontsize=4)
            if last_row & middle_in_row:
                axarr[i, j].set_xlabel('Percentage', labelpad=8, fontsize=4)
            if first_in_row:
                axarr[i, j].set_ylabel(trecs[i], labelpad=8, fontsize=4)

    # f.suptitle(r"RMS (left y-axis) & Kendall's $\tau$ (right y-axis)", fontsize=11)

    # Legend
    patch0 = mlines.Line2D([], [], color=colors[0], linewidth=0.4, marker=markers[0], markersize=2, label=labels[0])
    patch1 = mlines.Line2D([], [], color=colors[1], linewidth=0.4, marker=markers[1], markersize=2, label=labels[1])
    patch2 = mlines.Line2D([], [], color=colors[2], linewidth=0.4, marker=markers[2], markersize=2, label=labels[2])
    patch3 = mlines.Line2D([], [], color=colors[3], linewidth=0.4, marker=markers[3], markersize=2, label=labels[3])

    rms_line = mlines.Line2D([], [], linestyle='-', linewidth=0.4, color='black', label='$RMS$')
    tau_line = mlines.Line2D([], [], linestyle='--', linewidth=0.4, color='black', label=r'$\tau$')

    plt.legend(handles=[patch0, patch1, patch2, patch3, rms_line, tau_line], fontsize=4)

    plt.show()

    return


def plot_experiment_3_1():
    """Plot rms/tau wrt reusability for different model over all samples."""

    trecs = ['TREC-5', 'TREC-6', 'TREC-7', 'TREC-8', 'TREC-9', 'TREC-10', 'TREC-11']

    measures = ['ap', 'rp', 'p30']
    subtitles = ['MAP', 'RP', 'P@30']

    models = ['mtf', 'mab', 'importance', 'activewresap']
    labels = ['MTF', 'MAB', 'Stratif', 'Active']
    colors = ['#0088A8', '#B8860B', '#483D8B', '#DC143C']
    markers = ['^', 'v', '*', '.']

    stats = ['rms', 'tau']
    linestyles = ['-', '--']

    types = ['null']

    f, axarr = plt.subplots(len(trecs), len(measures))

    for i, trec_name in enumerate(trecs):
        for j, measure in enumerate(measures):
            ax_twin = axarr[i, j].twinx()
            for k, model in enumerate(models):
                df = pd.read_csv(os.path.join(os.path.join(EXP_DIR, trec_name), '{}.exp3.group.csv'.format(model)))
                for l, type in enumerate(types):
                    # Left y-axis
                    axarr[i, j].plot(df.loc[(df['measure'] == measure) & (df['type'] == type)].ix[:,'percentage'],
                                     df.loc[(df['measure'] == measure) & (df['type'] == type)].ix[:, stats[0]],
                                     linestyle=linestyles[0], label=model, linewidth=0.4, color=colors[k], marker=markers[k], markersize=2)

                    # Right y-axis
                    ax_twin.plot(df.loc[(df['measure'] == measure) & (df['type'] == type)].ix[:,'percentage'],
                                 df.loc[(df['measure'] == measure) & (df['type'] == type)].ix[:, stats[1]],
                                 linestyle=linestyles[1], label=model, linewidth=0.4, color=colors[k], marker=markers[k], markersize=2)

            # Tick fontsize & spacing
            axarr[i, j].xaxis.set_tick_params(pad=2, labelsize=4)
            axarr[i, j].yaxis.set_tick_params(pad=2, labelsize=4)
            ax_twin.yaxis.set_tick_params(pad=2, labelsize=4)

            # Grid
            axarr[i, j].grid(b=True, which='major', color='gray', linestyle='-',
                             alpha=0.25, zorder=1, lw=0.2)

            first_row = (i == 0)
            last_row = (i == len(trecs)-1)
            first_in_row = (j == 0)
            middle_in_row = (j == int(len(measures)/2))

            if first_row:
                axarr[i, j].set_title(subtitles[j], fontsize=4)
            if last_row & middle_in_row:
                axarr[i, j].set_xlabel('Percentage', labelpad=8, fontsize=4)
            if first_in_row:
                axarr[i, j].set_ylabel(trecs[i], labelpad=8, fontsize=4)

    # f.suptitle(r"RMS (left y-axis) & Kendall's $\tau$ (right y-axis)", fontsize=11)

    # Legend
    patch0 = mlines.Line2D([], [], color=colors[0], linewidth=0.4, marker=markers[0], markersize=2, label=labels[0])
    patch1 = mlines.Line2D([], [], color=colors[1], linewidth=0.4, marker=markers[1], markersize=2, label=labels[1])
    patch2 = mlines.Line2D([], [], color=colors[2], linewidth=0.4, marker=markers[2], markersize=2, label=labels[2])
    patch3 = mlines.Line2D([], [], color=colors[3], linewidth=0.4, marker=markers[3], markersize=2, label=labels[3])

    rms_line = mlines.Line2D([], [], linestyle='-', linewidth=0.4, color='black', label='$RMS$')
    tau_line = mlines.Line2D([], [], linestyle='--', linewidth=0.4, color='black', label=r'$\tau$')

    plt.legend(handles=[patch0, patch1, patch2, patch3, rms_line, tau_line], fontsize=4)

    plt.show()
    return


if __name__ == '__main__':
    # plot_dist_doc()
    # tbl_statistics_test_collection()

    # plot_experiment_1_1()
    # plot_experiment_1_2()
    # hypothesis_test_experiment_1_2()

    # plot_experiment_2_1()

    # plot_experiment_3_1()

    pass
