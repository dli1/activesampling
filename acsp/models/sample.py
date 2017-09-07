# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import csv
import numpy as np
import pandas as pd

from acsp.com.common import *
from acsp.com.utils import *
from acsp.models.active import ActiveModel
from acsp.models.basemodel import APDist, BoltzmannDist
from acsp.models.importance import ImportanceModel
from acsp.models.mtf import MTFModel
from acsp.models.mab import MABModel
from acsp.pre.preprocess import Preprocess


def unit_sample_procedure(trec_name, model, percentage, index, batch_num=BATCH_NUM):
    """Sampling and estimating, unit procedure."""

    # average estimator and answer over all topics
    array_ave_estimator = None
    array_ave_answer = None

    # get sysrun_name
    p = Preprocess(trec_name, str(DICT_TREC_TOPIC[trec_name][0]))
    list_sysrun_name = p.get_list_sysrun_name()

    # sampling and estimating for  all topics
    for topic_id in DICT_TREC_TOPIC[trec_name]:

        # the original data are large, thus only load data by topic id each time
        pre = Preprocess(trec_name, str(topic_id))
        budget_num = int(round(percentage * pre.get_unique_doc_num()))

        # the answer
        list_answer = pre.calculate_metrics()

        # the estimator
        array_doc = pre.get_doc_table()
        array_rel_answer = pre.get_rel_table()

        if 'activewr' == model:  # active sampling with replacement, exact ap as sampling distribution
            sm = APDist()
            md = ActiveModel(array_doc=array_doc, array_rel_answer=array_rel_answer, sample_model=sm)
            list_estimator = md.main(budget_num=budget_num, batch_num=batch_num,
                                     sampling_type=SAMPLING_TYPE_SAMPLE_WR, init_depth_k=0, weight_type=WEIGHT_TYPE_AP)

        elif 'activewresap' == model:  # active sampling with replacement, estimated ap as sampling distribution
            sm = APDist()
            md = ActiveModel(array_doc=array_doc, array_rel_answer=array_rel_answer, sample_model=sm)
            list_estimator = md.main(budget_num=budget_num, batch_num=batch_num,
                                     sampling_type=SAMPLING_TYPE_SAMPLE_WR, init_depth_k=0, weight_type=WEIGHT_TYPE_ESAP)

        elif 'activewor' == model:  # active sampling without replacement, exact ap as sampling distribution
            sm = APDist()
            md = ActiveModel(array_doc=array_doc, array_rel_answer=array_rel_answer, sample_model=sm)
            list_estimator = md.main(budget_num=budget_num, batch_num=None,
                                     sampling_type=SAMPLING_TYPE_SAMPLE_WOR, init_depth_k=0, weight_type=WEIGHT_TYPE_AP)

        elif 'mtf' == model:
            md = MTFModel(array_doc, array_rel_answer)
            list_estimator = md.main(budget_num)

        elif 'mab' == model:
            md = MABModel(array_doc, array_rel_answer, trec_name, topic_id)
            list_estimator = md.main(budget_num, 'complete', index)

        elif 'importance' == model:
            sm = APDist()
            md = ImportanceModel(array_doc, array_rel_answer, sm)
            list_estimator = md.main(budget_num)
        else:
            raise ValueError('Model {} is not implemented.'.format(model))

        # add estimator/answer into the array_ave_estimator/answer
        if array_ave_estimator is None:
            array_ave_estimator = np.array(list_estimator, dtype=float)
            array_ave_answer = np.array(list_answer, dtype=float)
        else:
            array_ave_estimator += np.array(list_estimator, dtype=float)
            array_ave_answer += np.array(list_answer, dtype=float)

    # average estimator/answer
    array_ave_estimator /= len(DICT_TREC_TOPIC[trec_name])
    array_ave_answer /= len(DICT_TREC_TOPIC[trec_name])

    return array_ave_estimator, array_ave_answer, list_sysrun_name


def multi_sample_procedures(trec_name, model, sample_index):
    """Sampling and estimating, multiple procedures."""

    print_time('start trec_name {}, model {}, sample_index {}'.format(trec_name, model, sample_index))

    # make sample_time_stamp string
    sample_dir = '{}{}'.format('sample', sample_index)

    # sample for different budget number
    for percentage in PERCENTAGES:

        # if file already exists, return
        percentage_dir = '{}{}'.format('percentage', int(percentage * 100))
        ret_dir = os.path.join(RESULT_DIR, trec_name, sample_dir, percentage_dir)
        file_name = '{}.csv'.format(model)
        if os.path.exists(os.path.join(ret_dir, file_name)):
            print('{} already exist.'.format(os.path.join(ret_dir, file_name)))
            return  # NOTE: return, not break (make sure the experiment within all percentages are finished one time).

        # else make dirs and file
        if not os.path.exists(ret_dir):
            os.makedirs(ret_dir)

        # run unit_sample_procedure
        array_ave_estimator, array_ave_answer, list_sysrun_name = unit_sample_procedure(trec_name, model,
                                                                                        percentage, sample_index)
        try:
            # write results
            f = codecs.open(os.path.join(ret_dir, file_name), 'w', encoding='utf-8')
            f_csv = csv.writer(f)
            f_csv.writerow(('system',
                            'estm_r', 'smpl_r', 'actu_r', 'smpl_doc', 'actu_doc',
                            'estm_ap', 'actu_ap', 'estm_rp', 'actu_rp', 'estm_p30', 'actu_p30', 'estm_dcg', 'actu_dcg',
                            'estm_rr', 'actu_rr', 'estm_rbp', 'actu_rbp', 'estm_bpref', 'actu_bpref'))

            for estimator, answer, sysrun_name in zip(array_ave_estimator, array_ave_answer, list_sysrun_name):
                f_csv.writerow((sysrun_name,
                                "{0:.4f}".format(estimator[0]),
                                "{0:.4f}".format(estimator[1]), "{0:.4f}".format(answer[1]),
                                "{0:.4f}".format(estimator[2]), "{0:.4f}".format(answer[2]),
                                "{0:.4f}".format(estimator[3]), "{0:.4f}".format(answer[3]),
                                "{0:.4f}".format(estimator[4]), "{0:.4f}".format(answer[4]),
                                "{0:.4f}".format(estimator[5]), "{0:.4f}".format(answer[5]),
                                "{0:.4f}".format(estimator[6]), "{0:.4f}".format(answer[6]),
                                "{0:.4f}".format(estimator[7]), "{0:.4f}".format(answer[7]),
                                "{0:.4f}".format(estimator[8]), "{0:.4f}".format(answer[8]),
                                "{0:.4f}".format(estimator[9]), "{0:.4f}".format(answer[9])))
            f.close()

        except Exception as e:
            os.remove(os.path.join(ret_dir, file_name))
            raise e
    return


def split_train_test(trec_name, topic_id, percentage, train_sysrun, test_sysrun):
    """Split data based on training runs and test runs."""

    pre = Preprocess(trec_name, str(topic_id))

    array_doc = pre.get_doc_table()
    array_rel_answer = pre.get_rel_table()
    sysruns = pre.get_list_sysrun_name()

    # train/test indices
    train_indices = [sysruns.index(run) for run in train_sysrun]
    test_indices = [sysruns.index(run) for run in test_sysrun]

    # train/test doc tables, train/test rel tables
    array_doc_train, array_doc_test = array_doc[train_indices], array_doc[test_indices]
    array_rel_answer_train, array_rel_answer_test = array_rel_answer[train_indices], array_rel_answer[test_indices]

    # train/test answer
    answers = np.array(pre.calculate_metrics())
    train_answer, test_answer = answers[train_indices], answers[test_indices]

    budget_num = int(round(percentage * pre.get_unique_doc_num(train_indices)))  # only count docs in training set

    return array_doc_train, array_doc_test, array_rel_answer_train, array_rel_answer_test, \
           train_answer, test_answer, budget_num


def unit_sample_train_test_procedure(trec_name, model, percentage, train_sysrun, test_sysrun, split_type, index,
                                     batch_num=BATCH_NUM, weight_type=WEIGHT_TYPE_AP):
    """Sampling on training set and estimating on training+test sets, unit procedure."""

    # average estimator over all topics
    ave_train_estimator = None
    ave_train_answer = None
    ave_test_estimator = None
    ave_test_answer = None

    # sampling and estimating for  all topics
    for topic in DICT_TREC_TOPIC[trec_name]:

        # split data based on train_indices and test_indices, get the answer
        array_doc_train, array_doc_test, \
            array_rel_answer_train, array_rel_answer_test, \
            train_answer, test_answer, \
            budget_num = split_train_test(trec_name, topic, percentage, train_sysrun, test_sysrun)

        # the estimator
        train_estimator = None
        test_estimator = None

        if 'activewr' == model:
            sm = APDist()
            md_train = ActiveModel(array_doc=array_doc_train, array_rel_answer=array_rel_answer_train, sample_model=sm)
            md_train.sample(budget_num=budget_num, batch_num=batch_num,
                            sampling_type=SAMPLING_TYPE_SAMPLE_WR, init_depth_k=0, weight_type=weight_type)
            train_estimator = md_train.estimate(sampling_type=SAMPLING_TYPE_SAMPLE_WR)

            md_test = ActiveModel(array_doc=array_doc_test, array_rel_answer=array_rel_answer_test, sample_model=sm)
            md_test.set_history_state(md_train.get_list_sampled_units())
            test_estimator = md_test.estimate(sampling_type=SAMPLING_TYPE_SAMPLE_WR)

        elif 'activewresap' == model:
            sm = APDist()
            md_train = ActiveModel(array_doc=array_doc_train, array_rel_answer=array_rel_answer_train, sample_model=sm)
            md_train.sample(budget_num=budget_num, batch_num=batch_num,
                            sampling_type=SAMPLING_TYPE_SAMPLE_WR, init_depth_k=0, weight_type=WEIGHT_TYPE_ESAP)
            train_estimator = md_train.estimate(sampling_type=SAMPLING_TYPE_SAMPLE_WR)

            md_test = ActiveModel(array_doc=array_doc_test, array_rel_answer=array_rel_answer_test, sample_model=sm)
            md_test.set_history_state(md_train.get_list_sampled_units())
            test_estimator = md_test.estimate(sampling_type=SAMPLING_TYPE_SAMPLE_WR)

        elif 'activewor' == model:
            sm = APDist()
            md_train = ActiveModel(array_doc=array_doc_train, array_rel_answer=array_rel_answer_train, sample_model=sm)
            md_train.sample(budget_num=budget_num, batch_num=None,
                            sampling_type=SAMPLING_TYPE_SAMPLE_WOR, init_depth_k=0, weight_type=weight_type)
            train_estimator = md_train.estimate(sampling_type=SAMPLING_TYPE_SAMPLE_WOR)

            md_test = ActiveModel(array_doc=array_doc_test, array_rel_answer=array_rel_answer_test, sample_model=sm)
            md_test.set_history_state(md_train.get_list_sampled_units())
            test_estimator = md_test.estimate(sampling_type=SAMPLING_TYPE_SAMPLE_WOR)

        elif 'mtf' == model:
            md_train = MTFModel(array_doc_train, array_rel_answer_train)
            md_train.sample(budget_num)
            train_estimator = md_train.estimate()

            md_test = MTFModel(array_doc_test, array_rel_answer_test)
            md_test.set_history_state(md_train.get_list_sampled_units())
            test_estimator = md_test.estimate()

        elif 'mab' == model:
            md_train = MABModel(array_doc_train, array_rel_answer_train, trec_name, topic)
            md_train.sample(budget_num, split_type, index)
            train_estimator = md_train.estimate()

            md_test = MABModel(array_doc_test, array_rel_answer_test, trec_name, topic)
            md_test.set_history_state(md_train.get_list_sampled_units())
            test_estimator = md_test.estimate()

        elif 'importance' == model:
            sm = APDist()
            md_train = ImportanceModel(array_doc_train, array_rel_answer_train, sm)
            md_train.sample(budget_num)
            train_estimator = md_train.estimate()

            md_test = ImportanceModel(array_doc_test, array_rel_answer_test, sm)
            md_test.set_history_state(md_train.get_list_sampled_units())
            test_estimator = md_test.estimate()

        # add estimator/answer into the array_ave_estimator/answer
        if ave_train_estimator is None:
            ave_train_estimator = np.array(train_estimator, dtype=float)
            ave_train_answer = np.array(train_answer, dtype=float)
            ave_test_estimator = np.array(test_estimator, dtype=float)
            ave_test_answer = np.array(test_answer, dtype=float)
        else:
            ave_train_estimator += np.array(train_estimator, dtype=float)
            ave_train_answer += np.array(train_answer, dtype=float)
            ave_test_estimator += np.array(test_estimator, dtype=float)
            ave_test_answer += np.array(test_answer, dtype=float)

    # average estimator/answer
    ave_train_estimator /= len(DICT_TREC_TOPIC[trec_name])
    ave_train_answer /= len(DICT_TREC_TOPIC[trec_name])
    ave_test_estimator /= len(DICT_TREC_TOPIC[trec_name])
    ave_test_answer /= len(DICT_TREC_TOPIC[trec_name])

    return ave_train_estimator, ave_train_answer, ave_test_estimator, ave_test_answer


def multi_sample_train_test_procedures(trec_name, model, split_type, split_index):
    """Sampling on training set and estimating on training+test sets, multiple procedures."""

    print_time('start trec_name {}, model {}, split_index {}'.format(trec_name, model, split_index))

    # sample_time_stamp string
    split_dir = 'split_{}_{}'.format(split_type, split_index)

    # read indices
    fname = '{}.csv'.format(split_type)
    df = pd.read_csv(os.path.join(os.path.join(PICKLE_DIR, trec_name), fname))
    train_sysrun = df.loc[(df['index'] == int(split_index)) & (df['train_test'] == 'train')].ix[:, 'run_name'].values
    test_sysrun = df.loc[(df['index'] == int(split_index)) & (df['train_test'] == 'test')].ix[:, 'run_name'].values

    # all percentages
    for percentage in PERCENTAGES:

        # if file already exists return
        percentage_dir = '{}{}'.format('percentage', int(percentage * 100))
        ret_dir = os.path.join(RESULT_DIR, trec_name, split_dir, percentage_dir)
        file_name = '{}.csv'.format(model)

        if os.path.exists(os.path.join(ret_dir, file_name)):
            return

        # else make dirs and file
        if not os.path.exists(ret_dir):
            os.makedirs(ret_dir)

        # run model
        ave_train_estimator, ave_train_answer, ave_test_estimator, ave_test_answer = \
            unit_sample_train_test_procedure(trec_name, model, percentage, train_sysrun, test_sysrun, split_type,
                                             split_index)

        try:
            # write results to csv file
            f = codecs.open(os.path.join(ret_dir, file_name), 'w', encoding='utf-8')
            f_csv = csv.writer(f)
            f_csv.writerow(('system', 'type',
                            'estm_r', 'smpl_r', 'actu_r', 'smpl_doc', 'actu_doc',
                            'estm_ap', 'actu_ap', 'estm_rp', 'actu_rp', 'estm_p30', 'actu_p30', 'estm_dcg', 'actu_dcg',
                            'estm_rr', 'actu_rr', 'estm_rbp', 'actu_rbp', 'estm_bpref', 'actu_bpref'))

            for ave_estimator, ave_answer, sysrun, mtype in [[ave_train_estimator, ave_train_answer, train_sysrun, 'train'],
                                                             [ave_test_estimator, ave_test_answer, test_sysrun, 'test']]:
                for estimator, answer, sysrun_name in zip(ave_estimator, ave_answer, sysrun):
                    f_csv.writerow((sysrun_name, mtype,
                                    "{0:.4f}".format(estimator[0]),
                                    "{0:.4f}".format(estimator[1]), "{0:.4f}".format(answer[1]),
                                    "{0:.4f}".format(estimator[2]), "{0:.4f}".format(answer[2]),
                                    "{0:.4f}".format(estimator[3]), "{0:.4f}".format(answer[3]),
                                    "{0:.4f}".format(estimator[4]), "{0:.4f}".format(answer[4]),
                                    "{0:.4f}".format(estimator[5]), "{0:.4f}".format(answer[5]),
                                    "{0:.4f}".format(estimator[6]), "{0:.4f}".format(answer[6]),
                                    "{0:.4f}".format(estimator[7]), "{0:.4f}".format(answer[7]),
                                    "{0:.4f}".format(estimator[8]), "{0:.4f}".format(answer[8]),
                                    "{0:.4f}".format(estimator[9]), "{0:.4f}".format(answer[9])))
            f.close()

        except Exception as e:
            os.remove(os.path.join(ret_dir, file_name))
            raise e
    return


if __name__ == '__main__':
    pass
