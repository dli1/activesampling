# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd

from acsp.com.common import *
from acsp.com.utils import *
from acsp.pre.preprocess import Preprocess


def write_order_doc_collections():
    """Prepare the input data of MAB baseline: rank documents by score."""

    trecs = ['TREC-5', 'TREC-6', 'TREC-7', 'TREC-8', 'TREC-9', 'TREC-10', 'TREC-11']

    for trec_name in trecs:
        sysrun_dir = os.path.join(DATA_DIR, trec_name, DICT_TREC_TYPE[trec_name][0])
        new_sysrun_dir = os.path.join(ORDER_DATA_DIR, trec_name, DICT_TREC_TYPE[trec_name][0])

        if not os.path.exists(new_sysrun_dir):
            os.makedirs(new_sysrun_dir)

        # read system runs
        file_ids = get_file_ids(sysrun_dir)
        for run_id in file_ids:
            # clear file every time rewriting the ranked list of the runs
            os.remove(os.path.join(new_sysrun_dir, '{}'.format(run_id)))

            # write new ranked list of the runs
            f_dir = os.path.join(new_sysrun_dir, '{}'.format(run_id))
            df = pd.read_csv(os.path.join(sysrun_dir, run_id), sep='\s+',
                             names=['topic_id', 'q0', 'doc_id', 'rank', 'score', 'run_name'])
            for topic_id in df.ix[:, 'topic_id'].unique():
                df_doc_ids = df.loc[df["topic_id"] == int(topic_id)].sort_values(by=["rank"], ascending=True)
                df_doc_ids.to_csv(f_dir, header=False, index=False,
                                  na_rep='NaN', sep=' ', encoding='utf-8', mode='a')

    return


def test_order_test_collection():
    """
    Test the sampling result of MAB: whether l1 == l2.
    :return:
    """
    for trec in ['TREC-5', 'TREC-6', 'TREC-7', 'TREC-8', 'TREC-9', 'TREC-10',  'TREC-11']:
        for topic_id in DICT_TREC_TOPIC[trec]:
            pre = Preprocess(trec, str(topic_id), 100)

            df = pd.read_csv(os.path.join(MIDWARE_DIR, 'midware_complete', trec, str(topic_id)))
            list_doc_id = list(df['doc'].values)

            l1 = pre.get_unique_doc_num()
            l2 = len(list_doc_id)
            if not (l1 == l2):
                print(trec, topic_id, l1, l2)

    return
