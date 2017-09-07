# coding=utf-8

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import random
from scipy import stats
from operator import itemgetter
from collections import defaultdict


from acsp.com.utils import *
from acsp.com.common import *

from acsp.models.basemodel import Model


class ImportanceModel(Model):
    """Importance sampling model"""

    def __init__(self, array_doc, array_rel_answer, sample_model):
        super(ImportanceModel, self).__init__(array_doc, array_rel_answer)
        self.sample_model = sample_model

    #****************************** Sampling methods ******************************#
    def update_weighted_doc_prob(self, budget_num):
        """Get the predicted document probability distribution."""

        list_predict_doc_prob = self.sample_model.predict_doc_prob(self.doc_num)

        # average the distribution over all systems
        dict_weighted_doc_prob = defaultdict(float)
        for i in range(self.system_num):
            for j in range(self.doc_num):
                doc_id = self.array_doc[i][j]
                dict_weighted_doc_prob[doc_id] = dict_weighted_doc_prob[doc_id] + list_predict_doc_prob[j]

        # normalize the distribution
        s = sum(dict_weighted_doc_prob.values())
        for key in dict_weighted_doc_prob.keys():
            if 0 == s:  # if there is only key with zero probability left, sample with equal prob
                dict_weighted_doc_prob[key] = 1.0 / float(len(dict_weighted_doc_prob.keys()))
            else:
                dict_weighted_doc_prob[key] /= float(s)

        # sort the distribution from high to low
        list_weighted_doc_prob = sorted(dict_weighted_doc_prob.items(), key=itemgetter(1), reverse=True)

        # chunk the distribution 
        list_chunked = chunks_by_element(list_weighted_doc_prob, budget_num)

        dict_chunked = defaultdict(dict)
        for i, chunk in enumerate(list_chunked):
            dict_chunked[i]['doc_ids'] = [doc_id for doc_id, prob in chunk]
            dict_chunked[i]['chunk_prob'] = sum([prob for doc_id, prob in chunk])

        return dict_chunked

    def update_all_docs(self, dict_chunked, budget_num):
        """
        Sample all the documents and update the rel table

        :param dict_chunked: dict
                {'chunc_id': {'doc_ids': list(), 'chunk_prob': float},
                ... }
        :param budget_num: int
                budget indicating documents to be sampled
        :return:
        """

        # sample chunks
        dist = stats.rv_discrete(name='importance',
                                 values=(dict_chunked.keys(),
                                         [dict_chunked[key]['chunk_prob'] for key in dict_chunked.keys()]))
        list_chunk_sample = list(dist.rvs(size=budget_num))

        # sample documents
        for key in dict_chunked.keys():
            sample_num = list_chunk_sample.count(key)  # count document number for each chunk
            list_doc_id = dict_chunked[key]['doc_ids']
            inclusion_prob = dict_chunked[key]['chunk_prob']
            list_sampled_doc = random.sample(list_doc_id, sample_num)  # simple wor with equal probability

            for sel_doc_id in list_sampled_doc:
                # project the result to all systems
                sel_doc_rel = self.update_rel(sel_doc_id)

                # save doc_id, doc_rel, dict_weighted_doc_prob
                self.list_sampled_units.append((sel_doc_id, sel_doc_rel, inclusion_prob))

        return

    def sample(self, budget_num):
        """Sample documents."""

        # update distribution
        dict_chunked = self.update_weighted_doc_prob(budget_num)

        # sample chunks and documents
        self.update_all_docs(dict_chunked, budget_num)

        # print('sampled doc, non repeated doc, iteration', len(self.list_sampled_units),
        #       len(set([doc for doc, rel, prob in self.list_sampled_units])), iteration)
        return

    ####################################Estimator methods########################################################
    def wr_pre_estimate(self):
        """Calculate inclusion probability, estimate relevant document number, irrelevant document number."""

        # remove duplicate documents and calculate inclusion probability
        list_inclu_prob = self.list_sampled_units

        # calculate population total
        total = 0
        for _, doc_rel, p in list_inclu_prob:
            total += float(doc_rel) / p

        # calculate number of total relevant docs
        esti_r = total

        # count the total number of sampled relevant docs and irrelevant docs
        smpl_r = 0
        smpl_n = 0
        for _, doc_rel, _ in self.list_sampled_units:
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
                total += float(doc_rel) / inclu_p / math.log(index + 2.0)  # log(rank+1)
        dcg = float(total) / float(ri)

        return ap, rp, p30, dcg

    def estimate(self):
        """Estimate rel doc, (real) rel doc, total doc, ap, rp, p30, dcg, rr, rbp, bpref. """

        list_estimator = []

        list_inclu_prob, esti_r, smpl_r, smpl_n = self.wr_pre_estimate()

        for system_index in range(self.system_num):
            ap, rp, p30, dcg = self.wr_estimate(list_inclu_prob, esti_r, system_index)
            rr, rbp, bpref = self.extended_estimate(smpl_r, smpl_n, system_index)
            list_estimator.append((esti_r, smpl_r, (smpl_r + smpl_n), ap, rp, p30, dcg, rr, rbp, bpref))

        return list_estimator

    #****************************** API ******************************#
    def main(self, iteration):
        """Main function """

        # sample
        self.sample(iteration)

        # estimate
        list_estimator = self.estimate()

        return list_estimator


if __name__ == '__main__':
    pass