# coding=utf-8

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from collections import defaultdict

from acsp.com.common import *
from acsp.com.utils import *


class Model(object):
    """
    Data structure for input data. It contains two types of tables:
    1. document table (run*depth matrix)
    S1 d_11, d_12, ...
    S2 d_21, d_22, ...
    S3 d_31, d_32, ...
    .
    .
    .

    2. relevance table (run*depth matrix)
    S1 r_11, r_12, ...
    S2 r_21, r_22, ...
    S3 r_31, r_32, ...
    .
    .
    .
    """
    def __init__(self, array_doc, array_rel_answer):
        # global state
        self.array_doc = np.array(array_doc, ndmin=2)
        self.array_rel_answer = np.array(array_rel_answer, ndmin=2)
        self.doc_num = self.array_doc.shape[1]  # column:  doc
        self.system_num = self.array_doc.shape[0]  # row:  sys

        # temporary state
        self.dict_weighted_doc_prob = defaultdict(float)
        self.sel_doc_id = None
        self.sel_doc_rel = None
        self.sel_sys = None

        # history state
        self.list_sampled_units = []
        self.array_rel = np.full_like(self.array_doc, fill_value=np.inf)

    #****************************** Sampling methods ******************************#
    def update_rel(self, sel_doc_id):
        """Query the relevance and project the result to all systems."""

        for i in range(self.system_num):
            for j in range(self.doc_num):
                if sel_doc_id == self.array_doc[i][j]:
                    self.array_rel[i][j] = self.array_rel_answer[i][j]
                    sel_doc_rel = self.array_rel_answer[i][j]
        return sel_doc_rel

    ##****************************** Estimator methods ******************************#
    def extended_estimate(self, smpl_r, smpl_n, system_index):
        """Estimate rr, rbp, bpref. """

        # reciprocal rank
        rr = 0

        p = 0.8
        rbp = 0

        bpref = 0

        for j in range(self.doc_num):
            if np.inf != self.array_rel[system_index][j]:
                rr += float(self.array_rel[system_index][j]) / float(j + 1)

                rbp += float(self.array_rel[system_index][j]) * p ** j

                if RELEVANT == self.array_rel[system_index][j]:
                    n_j = sum([1 for doc_rel in self.array_rel[system_index][:j] if NON_RELEVANT == doc_rel])

                    if 0 != n_j:
                        if 0 != smpl_r:
                            bpref += 1 - float(min(n_j, smpl_r)) / float(min(smpl_r, smpl_n))
                            if (1 - float(min(n_j, smpl_r)) / float(min(smpl_r, smpl_n))) < 0:
                                print('basemodel.extended_estimate: warning ', n_j, smpl_n, smpl_r)
                        else:  # smpl_r == 0 pass
                            print('basemodel.extended_estimate', j, doc_rel, n_j, smpl_r, smpl_n)
                    else:
                        bpref += 1

        rbp *= 1 - p

        if 0 != smpl_r:
            bpref *= 1.0 / smpl_r

        return rr, rbp, bpref

    #****************************** API ******************************#

    def sampled_doc(self):
        return len(set([doc for doc, rel, prob in self.list_sampled_units]))

    def sampled_rel(self):
        sum_rel = 0
        # remove duplicate documents and calculate inclusion probability
        list_local_doc = []
        for doc_id, doc_rel, _ in self.list_sampled_units:
            if doc_id not in list_local_doc:
                sum_rel += int(doc_rel)
                list_local_doc.append(doc_id)

        return sum_rel

    def get_list_sampled_units(self):
        return self.list_sampled_units

    def set_history_state(self, list_sampled_units):
        # sampled units
        self.list_sampled_units = list_sampled_units

        # array_rel
        for doc_id, doc_rel, p in list_sampled_units:
            for i in range(self.system_num):
                for j in range(self.doc_num):
                    if doc_id == self.array_doc[i][j]:
                        self.array_rel[i][j] = doc_rel
                        break

        return


class Distribution(object):
    """Sampling distribution"""

    def predict_doc_prob(self, z, list_sys_prob):
        """
        Generate probabilistic prediction.

        :param z: int
                number of documents in the document collection
        :param list_sys_prob: list
                system selection probability distribution
        :return: list of list
                document selection probability distribution for each system
        """
        raise NotImplementedError


class APDist(Distribution):
    def __init__(self):
        self.list_predicted_doc_prob = None

    def predict_doc_prob(self, z, list_sys_prob):

        # first time calling the method
        if self.list_predicted_doc_prob is None:
            self.list_predicted_doc_prob = []
            list_w = [math.log(z / r) for r in range(1, z + 1)]  # from 1 to z
            s = sum(list_w)
            list_normed_w = [j / s for j in list_w]
            for _ in list_sys_prob:
                self.list_predicted_doc_prob.append(list_normed_w)
        # not the first time, just skip as AP-prior is fixed during the sampling procedure
        else:
            pass

        return self.list_predicted_doc_prob


class BoltzmannDist(Distribution):
    def __init__(self):
        self.list_predicted_doc_prob = None

    def predict_doc_prob(self, z, list_sys_prob):
        self.list_predicted_doc_prob = []

        # scale
        mmax = 0.5
        mmin = 0.2
        taus = np.array(list_sys_prob)
        scale = 1 if 0 == (taus.max() - taus.min()) else (taus.max() - taus.min())
        taus_std = (taus - taus.min()) / scale
        taus_scaled = taus_std / (mmax - mmin) + mmin

        # generate distribution over docs for all systems
        for tau in taus_scaled:
            list_w = [float(1) / (r ** tau) for r in range(1, z + 1)]  # denominator starts from 1 to z
            s = sum(list_w)
            list_normed_w = [j / s for j in list_w]
            self.list_predicted_doc_prob.append(list_normed_w)

        return self.list_predicted_doc_prob
