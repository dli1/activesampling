# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import codecs
import pickle
import numpy as np
import pandas as pd
import xml.dom.minidom
from collections import defaultdict
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut

from acsp.com.common import *
from acsp.com.utils import *


class Preprocess(object):
    """Read system runs and qrels for a target topic."""

    def __init__(self, trec_name, topic_id, test_collection_depth=TEST_COLLECTION_DEPTH):
        """
        :param trec_name: str
        :param topic_id: str
        :param test_collection_depth: int
        """
        self.trec_name = trec_name
        self.topic_id = topic_id
        self.test_collection_depth = test_collection_depth

        self.pickle_dir = os.path.join(PICKLE_DIR, trec_name)
        self.sysrun_dir = os.path.join(DATA_DIR, trec_name, DICT_TREC_TYPE[trec_name][0])
        self.rel_dir = os.path.join(DATA_DIR, trec_name, DICT_TREC_TYPE[trec_name][1])
        self.midware_dir = os.path.join(MIDWARE_DIR, trec_name)

        # read data
        self.doc_table, self.rel_table, self.list_sysrun_name = \
            self.prepare_data(topic_id, self.pickle_dir, self.sysrun_dir, self.rel_dir, self.test_collection_depth)

    @staticmethod
    def read_sysrun(topic_id, sysrun_dir):

        dict_sysrun = dict()

        file_ids = get_file_ids(sysrun_dir)
        for run_id in file_ids:
            df = pd.read_csv(os.path.join(sysrun_dir, run_id), sep='\s+',
                             names=['topic_id', 'q0', 'doc_id', 'rank', 'score', 'run_name'])
            doc_ids = df.loc[df["topic_id"] == int(topic_id)].\
                          sort_values(by=["rank"], ascending=True).ix[:, 'doc_id'].values
            dict_sysrun[run_id.strip()] = list(doc_ids)

        return dict_sysrun

    @staticmethod
    def read_rel(topic_id, rel_dir):

        dict_rel = defaultdict(int)

        df = pd.read_csv(os.path.join(rel_dir, get_file_ids(rel_dir)[0]), sep=r'\s+',
                         names=['topic_id', '0', 'doc_id', 'rel'])
        df1 = df.loc[(df["topic_id"] == int(topic_id))]

        for index, row in df1.iterrows():
            doc_id = str(row['doc_id'])
            rel = int(row['rel'])
            dict_rel[doc_id] = RELEVANT if 0 != rel else NON_RELEVANT  # in case graded relevance judgement

        return dict_rel

    @staticmethod
    def make_tables(dict_sysrun, dict_rel, list_sysrun_name, depth):

        # make array of doc table
        doc_table = np.full((len(list_sysrun_name), depth), STR_NULL_DOC, dtype=basestring)

        # make array of rel table
        rel_table = np.full_like(doc_table, fill_value=NON_RELEVANT)

        for i, run_id in enumerate(list_sysrun_name):  # based on the order of self.list_sysrun_name!!!
            low_bound = min(len(dict_sysrun[run_id]), depth)
            for j in range(0, low_bound):
                doc_id = dict_sysrun[run_id][j]
                doc_table[i][j] = doc_id
                rel_table[i][j] = dict_rel[doc_id]

        return doc_table, rel_table

    @staticmethod
    def check_table(rel_table, doc_table):
        if NON_RELEVANT == set(rel_table.flatten()) or STR_NULL_DOC == set(doc_table.flatten()):
            raise ValueError('System runs or qrels loading error.')
        else:
            return

    def prepare_data(self, topic_id, pickle_dir, sysrun_dir, rel_dir, depth):
        """
        Read data from trec files or pickles (if not the first time)
        """

        # if not the first time, read from pickles to save loading time
        if os.path.exists(os.path.join(pickle_dir, str(topic_id))):
            with codecs.open(os.path.join(pickle_dir, str(topic_id)), 'r') as f:
                doc_table, rel_table, list_sysrun_name = pickle.load(f)
                self.check_table(rel_table, doc_table)

                if rel_table.shape[1] == depth:
                    return doc_table, rel_table, list_sysrun_name
                else:
                    pass  # if test_collection_depth is not consistent, pass to rewrite pickles

        # construct tables
        dict_sysrun = self.read_sysrun(topic_id, sysrun_dir)
        dict_rel = self.read_rel(topic_id, rel_dir)
        list_sysrun_name = dict_sysrun.keys()
        doc_table, rel_table = self.make_tables(dict_sysrun, dict_rel, list_sysrun_name, depth)
        self.check_table(rel_table, doc_table)

        # dump tables
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)
        with codecs.open(os.path.join(pickle_dir, str(topic_id)), 'w') as f:
            pickle.dump((doc_table, rel_table, list_sysrun_name), f)

        return doc_table, rel_table, list_sysrun_name

    def get_unique_doc_num(self, sysrun_indices=None):
        r, n = self._calculate_r_n(sysrun_indices)
        return r + n

    def relevance_rate(self):
        r, n = self._calculate_r_n()
        return r / float(r + n)

    def relevance_num(self):
        r, n = self._calculate_r_n()
        return r

    def get_doc_table(self):
        return self.doc_table

    def get_rel_table(self):
        return self.rel_table

    def get_list_sysrun_name(self):
        return self.list_sysrun_name

    def _calculate_r_n(self, sysrun_indices=None):

        # sysrun_indices is to count sysruns in training set
        if sysrun_indices is None:
            sysrun_indices = range(self.doc_table.shape[0])

        r = 0
        n = 0
        list_local_doc = []
        for i in range(self.doc_table.shape[0]):
            if i in sysrun_indices:
                for j in range(self.doc_table.shape[1]):
                    doc_id = self.doc_table[i][j]
                    if doc_id not in list_local_doc:
                        if RELEVANT == self.rel_table[i][j]:
                            r += 1
                        else:
                            n += 1
                    list_local_doc.append(doc_id)

        if STR_NULL_DOC in self.doc_table:
            n -= 1

        return r, n

    def calculate_metrics(self, r=None, n=None):
        """Calculate rel doc, (real) rel doc, total doc, ap, rp, p30, dcg, rr, rbp, bpref."""

        if r is None or n is None:
            r, n = self._calculate_r_n()  # for experiment
        else:
            r, n = r, n  # for test

        list_metrics = []
        for i, item in enumerate(self.rel_table):
            if 0 == r:
                list_metrics.append((0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
            else:
                ap = 0.0
                for j, doc_rel in enumerate(item):
                    ap += (sum(item[:j + 1]) / float(j + 1)) * doc_rel  # calculate precision at cutoff at i+1
                ap /= float(r)

                rp = sum(item[:r]) / float(r)

                p30 = sum(item[:CUTOFF_30]) / float(CUTOFF_30)

                dcg = sum(rel/math.log(index+2.0) for index, rel in enumerate(item)) / float(len(item))

                # reciprocal rank
                rr = sum([float(item[j]) / float(j + 1) for j in range(len(item))])

                # for j in range(len(item)):  # this is trec_eval's implementation
                #     if item[j] == 1:
                #         rr = float(item[j]) / float(j + 1)
                #         break

                # rbp
                p = 0.8
                rbp = (1 - p) * sum([float(item[j]) * p ** j for j in range(len(item))])

                # bpref
                bpref = 0.0
                for j, doc_rel in enumerate(item):
                    if RELEVANT == doc_rel:

                        n_j = sum([1 for rel in item[:j] if NON_RELEVANT == rel])
                        if 0 != n_j:
                            if 0 != r:
                                bpref += (1 - float(min(n_j, r)) / float(min(r, n)))
                            else:
                                print('preprocess.calculate_metrics', j, doc_rel, n_j, r, n)
                        else:
                            bpref += 1
                if 0 != r:
                    bpref /= r

                list_metrics.append((r, r, r + n, ap, rp, p30, dcg, rr, rbp, bpref))

        return list_metrics

    def stats_test_collection(self):
        """Statistics of the test collection."""
        dict_rel = self.read_rel(self.topic_id, self.rel_dir)
        return np.sum(dict_rel.values()),len(dict_rel.keys())


def make_groups(group_descp_path):
    """Make groups by the description file of participating teams in TREC tracks."""

    dict_run_group = dict()

    dom = xml.dom.minidom.parse(group_descp_path)
    root = dom.documentElement

    runs = root.getElementsByTagName('runs')
    for run in runs:
        run_name = 'input.' + get_tag_text(run, 'tag')
        org = get_tag_text(run, 'organization')
        dict_run_group[run_name] = org

    groups = list(set(dict_run_group.values()))
    for key in dict_run_group:
        dict_run_group[key] = groups.index(dict_run_group[key])

    return dict_run_group


def write_split_indices():
    """Save indices splitting train runs and test runs."""

    trecs = ['TREC-5', 'TREC-6', 'TREC-7', 'TREC-8', 'TREC-9', 'TREC-10', 'TREC-11']

    for trec_name in trecs:
        # system runs
        pre = Preprocess(trec_name, str(DICT_TREC_TOPIC[trec_name][0]))
        sysruns = pre.get_list_sysrun_name()
        x = np.array(sysruns)

        # groups
        dict_run_group = make_groups(os.path.join(DATA_DIR, trec_name, DICT_TREC_TYPE[trec_name][2], 'desc.xml'))
        groups = [dict_run_group[sysrun] for sysrun in sysruns]

        #****************************** 'leave-one-run-out' == split_type: ******************************#
        loo = LeaveOneOut()

        # create file
        f = codecs.open(os.path.join(os.path.join(PICKLE_DIR, trec_name), 'leave-one-run-out.csv'), 'w',
                        encoding='utf-8')
        f_csv = csv.writer(f)
        f_csv.writerow(('index', 'train_test', 'run_name'))

        # write indices
        indices = [(train_index, test_index) for train_index, test_index in loo.split(X=x)]
        for i, (train_index, test_index) in enumerate(indices):
            for run in x[train_index]:
                f_csv.writerow((i, 'train', run))
            for run in x[test_index]:
                f_csv.writerow((i, 'test', run))
        f.close()

        #****************************** 'leave-one-group-out' == split_type: ******************************#
        loo = LeaveOneGroupOut()

        # create file
        f = codecs.open(os.path.join(os.path.join(PICKLE_DIR, trec_name), 'leave-one-group-out.csv'), 'w',
                        encoding='utf-8')
        f_csv = csv.writer(f)
        f_csv.writerow(('index', 'train_test', 'run_name'))

        # write indices
        indices = [(train_index, test_index) for train_index, test_index in loo.split(X=x, groups=groups)]
        for i, (train_index, test_index) in enumerate(indices):
            for run in x[train_index]:
                f_csv.writerow((i, 'train', run))
            for run in x[test_index]:
                f_csv.writerow((i, 'test', run))
        f.close()

    return


if __name__ == '__main__':
    pass
