# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from acsp.models.eval import calculate_experiment_1_1, calculate_experiment_2_1, calculate_experiment_3_1
from acsp.models.sample import multi_sample_procedures, multi_sample_train_test_procedures
from acsp.pre.preprocess import write_split_indices

FLAGS = None


def main():
    """
    Main function
    """

    # print all parameter settings
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

    action = FLAGS.action
    experiment = FLAGS.experiment
    sample_index = FLAGS.sample_index
    split = FLAGS.split
    split_type = FLAGS.split_type
    trec = FLAGS.trec
    model = FLAGS.model

    # run the operation
    if 'preprocess' == action:
        write_split_indices()  # split indices for training set and test set

    elif 'sample' == action:
        if 'no' == split:
            multi_sample_procedures(trec, model, sample_index)
        elif 'yes' == split:
            multi_sample_train_test_procedures(trec, model, split_type, sample_index)

    elif 'evaluate' == action:
        if '1' == experiment:
            calculate_experiment_1_1()
        elif '2' == experiment:
            calculate_experiment_2_1()
        elif '3' == experiment():
            calculate_experiment_3_1()
    else:
        pass

    return


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='sample',
                        help='Choose action, preprocess: preprocessing of input data; sample: sample documents and estimate measures, evaluate: evaluate model performance.')
    parser.add_argument('--experiment', type=str, default='1',
                        help='Choose experiment, 1:bias and variance , 2: effectiveness, 3: re-usability.')

    parser.add_argument('--split', type=str, default='no',
                        help='Whether split data into training/test, no: not split, yes: split')
    parser.add_argument('--split_type', type=str, default='leave-one-group-out',
                        help='leave-one-group-out or leave-one-run-out.')

    parser.add_argument('--trec', type=str, default='TREC-5',
                        help='TREC Name used in data directory.')
    parser.add_argument('--model', type=str, default='mtf',
                        help='Choose models, mtf: move-to-front, importance: importance sampling, mab: multi-armed bandit, activewr: active sampling.')
    parser.add_argument('--sample_index', type=str, default='1',
                        help='Number indicating the repeated time of sampling and estimating procedure. e.g. 1, 2, 3, ...')

    FLAGS, unparsed = parser.parse_known_args()

    # run main function
    main()

    pass
