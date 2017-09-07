# coding=utf-8

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import math
import numpy as np
from scipy import stats
from functools import reduce
from collections import defaultdict

from acsp.com.common import *
from acsp.models.basemodel import Model


class ActiveModel(Model):
    """Active sampling"""

    def __init__(self, array_doc, array_rel_answer, sample_model):
        super(ActiveModel, self).__init__(array_doc, array_rel_answer)
        self.set_sampled_doc = set()
        self.sample_model = sample_model

    #****************************** Sampling methods ******************************#
    def _init_array_rel(self, init_depth_k):
        if 0 == init_depth_k:
            return

        for i in range(self.system_num):
            for j in range(init_depth_k):
                self.array_rel[i][j] = self.array_rel_answer[i][j]
        return

    def _update_sysrun_weight(self, weight_type, sampling_type, plot_flag=False):
        # the default is uniform distribution
        list_sys_prob = list(np.full((self.system_num,), fill_value=1.0 / self.system_num, dtype=float))
        alphas = np.full((self.system_num,), fill_value=NON_RELEVANT, dtype=float)

        # reciprocal of ranking
        if 0b0010 == 0b0010 & weight_type:
            for i in range(self.system_num):
                for j in range(self.doc_num):
                    if np.inf != self.array_rel[i][j]:
                        alphas[i] += float(self.array_rel[i][j]) / float(j + 1)
            list_sys_prob = list_sys_prob if 0 == sum(alphas) else list(alphas / sum(alphas))

            if 0b1 == 0b1 & weight_type:
                list_sys_prob = list(np.random.dirichlet(list_sys_prob, 1)[0])

        # proportional to ap
        elif 0b0100 == 0b0100 & weight_type:
            for i in range(self.system_num):
                for j in range(self.doc_num):
                    for k in range(j, self.doc_num):
                        if np.inf != self.array_rel[i][j]:
                            alphas[i] += float(self.array_rel[i][j]) / float(k + 1)
            list_sys_prob = list_sys_prob if 0 == sum(alphas) else list(alphas / sum(alphas))

            if 0b1 == 0b1 & weight_type:
                list_sys_prob = list(np.random.dirichlet(list_sys_prob, 1)[0])

        # esap
        elif 0b010000 == 0b010000 & weight_type:

            alphas = [estimator[3] for estimator in self.estimate(sampling_type)]
            list_sys_prob = list_sys_prob if 0 == sum(alphas) else list(alphas / sum(alphas))

            if 0b1 == 0b1 & weight_type:
                list_sys_prob = list(np.random.dirichlet(list_sys_prob, 1)[0])

        # bpref
        elif 0b1000 == 0b1000 & weight_type:
            r = 0
            n = 0
            list_local_doc = []
            for i in range(self.system_num):
                for j in range(self.doc_num):
                    if np.inf != self.array_rel[i][j]:
                        doc_id = self.array_doc[i][j]
                        if doc_id not in list_local_doc:
                            if RELEVANT == self.array_rel[i][j]:
                                r += 1
                            else:
                                n += 1
                        list_local_doc.append(doc_id)

            for i in range(self.system_num):
                for j in range(self.doc_num):
                    if np.inf != self.array_rel[i][j]:
                        # subset where the rank position of any unit is less than j.
                        n_j = sum([1 for doc_rel in self.array_rel[i][:j] if NON_RELEVANT == doc_rel])
                        if 0 != n_j:
                            if 0 != r:
                                alphas[i] += (1 - float(min(n_j, r)) / float(min(r, n)))
                            else:  # r == 0 pass
                                # print('active._update_sysrun_weight', i, j, n_j, r, n)
                                pass
                        else:
                            alphas[i] += 1
            list_sys_prob = list_sys_prob if 0 == sum(alphas) else list(alphas / sum(alphas))

            if 0b1 == 0b1 & weight_type:
                list_sys_prob = list(np.random.dirichlet(list_sys_prob, 1)[0])

        if plot_flag is True:
            print(list_sys_prob)

        return list_sys_prob

    def update_weighted_doc_prob(self, list_sys_prob, list_predicted_doc_prob):
        """
        Calculate weighted document selection probability.

        At each round, there are always hundreds of documents for which the probability is zero.

        :param list_sys_prob: list
                probability of selecting a system, calculated by normalizing system weights
        :param list_predicted_doc_prob: list
                probability of selecting a document in the target rank
        :return: dict
                d[doc] = prob
        """

        dict_weighted_doc_prob = defaultdict(float)  # must clear it!!!

        for i in range(self.system_num):
            for j in range(self.doc_num):
                doc = self.array_doc[i][j]
                p = list_sys_prob[i] * list_predicted_doc_prob[i][j]
                dict_weighted_doc_prob[doc] += p

        return dict_weighted_doc_prob

    def update_next_docs(self, dict_weighted_doc_prob, sampling_type, batch_num, budget_num):
        """
        Sample the next documents and update the rel table

        :param dict_weighted_doc_prob:
        :param sampling_type: int
                3 sampling types are available: argmax, wr, wor
        :param batch_num: int
                number of documents sampled every round
        :param budget_num: int
                budget indicating documents to be sampled

        :return:
        """
        list_doc = []

        # 1. sample
        if SAMPLING_TYPE_ARGMAX == sampling_type:
            # get the key with the maximum value.
            # if there are more than one keys having the maximum value, only return the first one.
            doc = max(dict_weighted_doc_prob.items(), key=lambda x: x[1])[0]
            list_doc.append(doc)

        elif SAMPLING_TYPE_SAMPLE_WR == sampling_type:
            if budget_num < batch_num:
                num_sample_per_batch = budget_num
                print('Warning: budget_num: {} < batch_num: {} !'.format(budget_num, batch_num))
            else:
                num_sample_per_batch = int(budget_num / batch_num)

            dist = stats.rv_discrete(name='wr', values=(range(len(dict_weighted_doc_prob.keys())),
                                                         dict_weighted_doc_prob.values()))
            indexes = dist.rvs(size=num_sample_per_batch)
            list_doc = [dict_weighted_doc_prob.keys()[inx] for inx in np.nditer(indexes)]

        elif SAMPLING_TYPE_SAMPLE_WOR == sampling_type:
            # delete the keys of docs already sampled
            for doc, rel, prob in self.list_sampled_units:
                del (dict_weighted_doc_prob[doc])

            # normalize the weights
            s = sum(dict_weighted_doc_prob.values())
            for key in dict_weighted_doc_prob.keys():
                if 0 == s:  # if there is only key with zero probability left, sample with equal prob
                    dict_weighted_doc_prob[key] = 1 / float(len(dict_weighted_doc_prob.keys()))
                else:
                    dict_weighted_doc_prob[key] /= float(s)

            dist = stats.rv_discrete(name='wor',
                                     values=(range(len(dict_weighted_doc_prob.keys())),
                                             dict_weighted_doc_prob.values()))
            inx = dist.rvs(size=1)[0]
            doc = dict_weighted_doc_prob.keys()[inx]
            list_doc.append(doc)

        # 2. assess: updating the rel table
        for sel_doc_id in list_doc:

            # no budget
            if not (len(self.set_sampled_doc) < budget_num):
                break

            # project the result to all systems
            sel_doc_rel = self.update_rel(sel_doc_id)

            # save doc_id, doc_rel, dict_weighted_doc_prob
            self.list_sampled_units.append((sel_doc_id, sel_doc_rel, dict_weighted_doc_prob))
            self.set_sampled_doc.add(sel_doc_id)

            # update temporal state
            self.dict_weighted_doc_prob = dict_weighted_doc_prob
            self.sel_doc_id = sel_doc_id
            self.sel_doc_rel = sel_doc_rel

        return

    def sample(self, budget_num, sampling_type, batch_num, init_depth_k, weight_type):
        """Sample documents."""

        # initialize rel , sample top k documents for training sample model
        self._init_array_rel(init_depth_k)

        # update system weights
        list_sys_prob = self._update_sysrun_weight(weight_type, sampling_type)

        # sample
        count = 50
        while True:  # sampling time is not predictable for wr, therefore use while
            # update dict_weighted_doc_prob
            list_predicted_doc_prob = self.sample_model.predict_doc_prob(self.doc_num, list_sys_prob)
            dict_weighted_doc_prob = self.update_weighted_doc_prob(list_sys_prob, list_predicted_doc_prob)

            # sample the next documents and update the rel table
            self.update_next_docs(dict_weighted_doc_prob, sampling_type, batch_num, budget_num)

            # update system probabilities
            list_sys_prob = self._update_sysrun_weight(weight_type, sampling_type)

            # no budget
            if not (len(self.set_sampled_doc) < budget_num):
                break

            # loop too many times
            count -= 1
            if count == 0:
                break

        # print('sampled doc, non repeated doc, budget_num', len(self.list_sampled_units),
        #       len(set([doc for doc, rel, prob in self.list_sampled_units])), budget_num)
        return

    #****************************** Estimator methods ******************************#
    def wor_pre_estimate(self):

        list_condi_prob = []
        esti_r = 0
        smpl_r = 0
        smpl_n = 0

        return list_condi_prob, esti_r, smpl_r, smpl_n

    def wor_estimate(self, list_condi_prob, r, system_index):
        """Estimate ap, rp, p30 for wor sampling. """

        ap = 0
        rp = 0
        p30 = 0
        dcg = 0

        return ap, rp, p30, dcg

    def wr_pre_estimate(self):
        """Calculate inclusion probability, estimate relevant document number, irrelevant document number."""

        # remove duplicate documents and calculate inclusion probability
        list_local_doc = []
        list_inclu_prob = []
        for doc_id, doc_rel, _ in self.list_sampled_units:
            if doc_id not in list_local_doc:
                list_prob = [(1.0 - dict_weighted_doc_prob[doc_id])
                             for _, _, dict_weighted_doc_prob in self.list_sampled_units]
                inclu_p = 1.0 - reduce(lambda x, y: x * y, list_prob)
                list_inclu_prob.append((doc_id, doc_rel, inclu_p))
                list_local_doc.append(doc_id)

        # estimate the total number of sampled relevant docs
        esti_r = 0
        for _, doc_rel, p in list_inclu_prob:
            esti_r += float(doc_rel) / p

        # count the total number of sampled relevant docs and irrelevant docs
        smpl_r = 0
        smpl_n = 0
        for _, doc_rel, p in list_inclu_prob:
            if 1 == float(doc_rel):
                smpl_r += 1
            else:
                smpl_n += 1

        return list_inclu_prob, esti_r, smpl_r, smpl_n

    def wr_estimate(self, list_inclu_prob, r, system_index):
        """Estimate ap, rp, p30, dcg for wr sampling"""

        if 0 == r:
            return 0, 0, 0

        # get the ranking list of a system run
        list_doc = list(self.array_doc[system_index])

        list_inclu_prob = [(doc_id, doc_rel, condi_p) for doc_id, doc_rel, condi_p in
                           list_inclu_prob if doc_id in list_doc]

        # calculate pc(r(i))
        list_precision = []
        for ddoc_id, ddoc_rel, dinclu_p in list_inclu_prob:
            ri = list_doc.index(ddoc_id) + 1.0

            # calculate pc
            total = 0
            for doc_id, doc_rel, inclu_p in list_inclu_prob:
                if (list_doc.index(doc_id)) < ri:
                    total += float(doc_rel) / inclu_p
            pc = float(total) / float(ri)
            list_precision.append((ddoc_id, ddoc_rel, dinclu_p, pc))

        # calculate ap
        total = 0
        for _, doc_rel, inclu_p, pc in list_precision:
            y = doc_rel * pc
            total += float(y) / inclu_p
        ap = total / float(r)

        # calculate rp
        total = 0
        for doc_id, doc_rel, inclu_p in list_inclu_prob:
            if (list_doc.index(doc_id)) < r:
                total += float(doc_rel) / inclu_p
        rp = float(total) / float(r)

        # calculate p30
        total = 0
        for doc_id, doc_rel, inclu_p in list_inclu_prob:
            if (list_doc.index(doc_id)) < CUTOFF_30:
                total += float(doc_rel) / inclu_p
        p30 = float(total) / float(CUTOFF_30)

        # calculate ncg
        total = 0
        ri = len(list_doc)
        for doc_id, doc_rel, inclu_p in list_inclu_prob:
            index = list_doc.index(doc_id)
            if index < ri:
                total += float(doc_rel) / inclu_p / math.log(index+2.0)  # log(rank+1)
        dcg = float(total) / float(ri)

        return ap, rp, p30, dcg

    def estimate(self, sampling_type):
        """Estimate rel doc, (real) rel doc, total doc, ap, rp, p30, dcg, rr, rbp, bpref. """

        list_estimator = []

        if SAMPLING_TYPE_SAMPLE_WOR == sampling_type:
            list_prob, esti_r, smpl_r, smpl_n = self.wor_pre_estimate()
            for system_index in range(self.system_num):
                ap, rp, p30, dcg = self.wor_estimate(list_prob, esti_r, system_index)
                rr, rbp, bpref = self.extended_estimate(smpl_r, smpl_n, system_index)
                list_estimator.append((esti_r, smpl_r, (smpl_r + smpl_n), ap, rp, p30, dcg, rr, rbp, bpref))

        elif SAMPLING_TYPE_SAMPLE_WR == sampling_type:
            list_prob, esti_r, smpl_r, smpl_n = self.wr_pre_estimate()
            for system_index in range(self.system_num):
                ap, rp, p30, dcg = self.wr_estimate(list_prob, esti_r, system_index)
                rr, rbp, bpref = self.extended_estimate(smpl_r, smpl_n, system_index)
                list_estimator.append((esti_r, smpl_r, (smpl_r + smpl_n), ap, rp, p30, dcg, rr, rbp, bpref))
        else:
            raise ValueError('Sampling is not WR or WOR!')

        return list_estimator

    #****************************** API ******************************#
    def main(self, budget_num, sampling_type, batch_num, init_depth_k, weight_type):
        """Main function"""

        # sample
        self.sample(budget_num, sampling_type, batch_num, init_depth_k, weight_type)

        # estimate
        list_estimator = self.estimate(sampling_type)

        return list_estimator


if __name__ == '__main__':
    pass

