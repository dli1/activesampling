# coding=utf-8

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import copy
import math
import numpy as np
from operator import itemgetter

from acsp.com.common import *
from acsp.models.basemodel import Model


class MTFModel(Model):
    """Move-to-front model"""

    def __init__(self, array_doc, array_rel_answer):
        super(MTFModel, self).__init__(array_doc, array_rel_answer)
        self.set_sampled_doc = set()

    #****************************** Sampling methods ******************************#
    def sample(self, budget_num):
        """Sample documents."""

        # initially, all runs have the maximum priority, which is set to a value equal to pool_depth
        dict_queried_doc_rank = {}
        dict_priority = {}
        for key in range(self.system_num):
            dict_priority[key] = self.doc_num
            dict_queried_doc_rank[key] = 0

        # initially, uniformly sample a system
        system_index = np.random.randint(0, self.system_num, 1)[0]

        while True:

            for doc_rank in range(dict_queried_doc_rank[system_index], self.doc_num):

                if self.array_rel[system_index][doc_rank] == np.inf:
                    # sample one document
                    sel_doc_id = self.array_doc[system_index][doc_rank]

                    # project the result to all systems
                    sel_doc_rel = self.update_rel(sel_doc_id)

                    # save doc_id, doc_rel
                    self.list_sampled_units.append((sel_doc_id, sel_doc_rel, 0))
                    self.set_sampled_doc.add(sel_doc_id)

                    # update temporal state
                    self.sel_doc_id = sel_doc_id
                    self.sel_doc_rel = sel_doc_rel
                    self.sel_sys = system_index

                    dict_queried_doc_rank[system_index] = doc_rank + 1

                    # if fully judged, force jump
                    if not (self.doc_num > dict_queried_doc_rank[system_index]):
                        dict_priority.pop(system_index)
                        dict_queried_doc_rank.pop(system_index)
                        if {} != dict_priority:
                            system_index = sorted(dict_priority.items(), key=itemgetter(1), reverse=True)[0][0]
                            break

                    # if not relevant, query another system
                    if NON_RELEVANT == sel_doc_rel:
                        # select the index for the new system
                        for index, priority in sorted(dict_priority.items(), key=itemgetter(1), reverse=True):
                            if index != system_index:
                                # current system subtract 1 when jump
                                dict_priority[system_index] = dict_priority.get(system_index, self.doc_num) - 1
                                system_index = index  # new system
                                break
                        break  # jump to another system

                else:
                    dict_queried_doc_rank[system_index] = doc_rank + 1

                    # if fully judged, force jump
                    if not (self.doc_num > dict_queried_doc_rank[system_index]):
                        del dict_priority[system_index]
                        del dict_queried_doc_rank[system_index]
                        system_index = sorted(dict_priority.items(), key=itemgetter(1), reverse=True)[0][0]
                        break

            # no budget
            if not (len(self.set_sampled_doc) < budget_num):
                break

        # print('sampled doc, non repeated doc, num_sample_per_batch', len(self.list_sampled_units),
        #       len(set([doc for doc, rel, prob in self.list_sampled_units])), iteration)
        return

    #****************************** Estimator methods ******************************#
    def pre_estimate(self):
        """Calculate relevant/non-relevant document number. """

        # remove duplicate documents
        list_local_doc = []
        list_sampled_units = []
        for doc_id, doc_rel, p in self.list_sampled_units:
            if doc_id not in list_local_doc:
                list_sampled_units.append((doc_id, doc_rel, p))
                list_local_doc.append(doc_id)
        self.list_sampled_units = list_sampled_units

        # count the total number of sampled relevant docs and irrelevant docs
        smpl_r = 0
        smpl_n = 0
        for _, doc_rel, _ in self.list_sampled_units:
            if 1 == float(doc_rel):
                smpl_r += 1
            else:
                smpl_n += 1

        return smpl_r, smpl_n

    def basic_estimate(self, r, system_index):
        """Estimate ap, rp, p30, dcg. """

        if 0 == r:
            return 0, 0, 0

        item = copy.deepcopy(self.array_rel[system_index])
        item[np.inf == item] = NON_RELEVANT  # suppose the unjudged documents are not relevant

        ap = 0
        for i, doc_rel in enumerate(item):
            ap += (item[:i + 1].sum() / float(i + 1)) * doc_rel  # calculate precision at cutoff at i+1
        ap /= float(r)

        rp = sum(item[:r]) / float(r)

        p30 = sum(item[:CUTOFF_30]) / float(CUTOFF_30)

        dcg = sum(rel/math.log(index+2.0) for index, rel in enumerate(item)) / float(len(item))

        return ap, rp, p30, dcg

    def estimate(self):
        """Estimate rel doc, (real) rel doc, total doc, ap, rp, p30, dcg, rr, rbp, bpref. """

        (smpl_r, smpl_n) = self.pre_estimate()

        list_estimator = []
        for system_index in range(self.system_num):

            (ap, rp, p30, dcg) = self.basic_estimate(smpl_r, system_index)
            (rr, rbp, bpref) = self.extended_estimate(smpl_r, smpl_n, system_index)
            list_estimator.append((smpl_r, smpl_r, (smpl_r + smpl_n), ap, rp, p30, dcg, rr, rbp, bpref))

        return list_estimator

    #****************************** API ******************************#
    def main(self, budget_num):
        # sample
        self.sample(budget_num)

        # estimate
        list_estimator = self.estimate()

        return list_estimator
