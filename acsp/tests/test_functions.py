# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd

from acsp.com.common import *
from acsp.com.utils import *
from acsp.pre.trec_eval import TrecEval
from acsp.pre.preprocess import Preprocess


def test_metrics(trec_name):
    """Test method Preprocess.calculate_metrics()"""

    for topic_id in DICT_TREC_TOPIC[trec_name]:
        # acsp answer
        pre = Preprocess(trec_name, str(topic_id))
        dict_rel = pre.read_rel(topic_id, pre.rel_dir)
        r = 0
        n = 0
        for key in dict_rel:
            if RELEVANT == dict_rel[key]:
                r += 1
            else:
                n += 1
        list_metrics = pre.calculate_metrics(r, n)

        # trec_eval answer
        ta = TrecEval(trec_name)
        list_answer = ta.get_answer(str(topic_id), pre.list_sysrun_name)

        # compare results
        print('topic,  system,  trec_eval answer,  acsp answer')
        for name in pre.list_sysrun_name:
            index = pre.list_sysrun_name.index(name)
            anwser = list_answer[index]
            metrics = list_metrics[index]
            print(' {} {} {} {}'.format(topic_id, name, anwser, metrics))

    return

