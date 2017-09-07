# coding=utf-8

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import pandas as pd

from acsp.com.common import *
from acsp.com.utils import *

from acsp.models.mtf import MTFModel


class MABModel(MTFModel):
    """Multi-armed bandit model"""

    def __init__(self, array_doc, array_rel_answer, trec_name, query_id):
        super(MABModel, self).__init__(array_doc, array_rel_answer)
        self.trec_name = trec_name
        self.query_id = query_id

    #****************************** Sampling methods ******************************#
    def sample(self, budget_num, split_type, sample_index):
        """
        Sample documents

        :param budget_num: int
        :param split_type: str
                complete: put all runs in the pool , leave-one-group-out: leave one group runs out of the pool,
                leave-one-run-out: leave one run out the of pool.
        :param sample_index: str
                Number indicating the repeated time of sampling and estimating procedure.
        :return:
        """
        # read sampled documents
        midware_dir = 'midware_{}_{}'.format(split_type, sample_index)
        df = pd.read_csv(os.path.join(MIDWARE_DIR, self.trec_name, midware_dir, str(self.query_id)))

        # cut off the result list by budge_num
        list_doc_id = list(df['doc'].values)[:budget_num]

        for sel_doc_id in list_doc_id:

            # project the result to all systems
            sel_doc_rel = self.update_rel(sel_doc_id)

            # save doc_id, doc_rel
            self.list_sampled_units.append((sel_doc_id, sel_doc_rel, 0))

        return
    #************* Estimator methods are the same with MTFModel. *************#


    #****************************** API ******************************#
    def main(self, budget_num, split_type, sample_index):
        # sample
        self.sample(budget_num, split_type, sample_index)

        # estimate
        list_metrics = self.estimate()

        return list_metrics
