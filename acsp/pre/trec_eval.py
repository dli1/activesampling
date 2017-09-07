# coding=utf-8

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import subprocess
from collections import defaultdict

from acsp.com.common import *
from acsp.com.utils import *


class EvaluationResult(object):
    """An object that stores the results output by trec_eval."""

    eval_fields = {
        "num_q": int,
        "num_ret": int,
        "num_rel": int,
        "num_rel_ret": int,
        "map": float,
        "Rprec": float,
        "bpref": float,
        "recip_rank": float,
        "P_10": float,
        "P_30": float,
        "P_100": float}

    def __init__(self, rc):
        """Initializes from a file which contains the output from trec_eval."""

        self.runid = ""
        self.results = {}
        self.queries = {}

        # with open(filepath, 'r') acsp f:
        for line in rc.split('\n'):

            if line.strip() == '':
                continue

            (field, query, value) = line.split()

            if query == "all":  # accumulated results over all queries
                if field == "runid":
                    self.runid = value
                else:
                    self.parse_field(field, value, self.results)
            else:  # query is a number
                if query not in self.queries:
                    self.queries[query] = {}
                self.parse_field(field, value, self.queries[query])

    def parse_field(self, field, value, target):
        """Parses the value of a field and puts it in target[field]."""

        field_types = self.__class__.eval_fields

        if field in field_types:
            target[field] = field_types[field](value)  # convert str type to target type
        else:
            pass

    def get_total_measure_score(self, field):
        """Get average measure for all queries."""
        return self.results[field]

    def get_measure_score_by_query(self, field, query):
        """
        Get measure by query.
        """
        if query in self.queries.keys():
            return self.queries[query].get(field, 0)
        else:
            return 0


class TrecEval(object):
    """Wrapper of trec_eval."""

    def __init__(self, trec_name):
        self.dict_answer = self.calculate_metrics(trec_name=trec_name)

    @staticmethod
    def trec_eval(path_qrels_file, path_query_rt_file, path_trec_eval=TREC_EVAL_EXCUTE):
        """Call trec_eval via subprocess."""

        # call trec_eval in command
        pipe = subprocess.Popen([path_trec_eval, '-q', path_qrels_file, path_query_rt_file], stdout=subprocess.PIPE)
        try:
            stdout, stderr = pipe.communicate()
        except subprocess.CalledProcessError as e:
            raise e

        # format result
        rc_str = stdout.decode()
        eval_result = EvaluationResult(rc_str)

        return eval_result

    def calculate_metrics(self, trec_name):
        """Calculate rel doc, rel doc, total doc, ap, rp, p30, rr, rbp, bpref."""

        # qrel
        rel_dir = os.path.join(DATA_DIR, trec_name, DICT_TREC_TYPE[trec_name][1])
        path_qrels_file = os.path.join(rel_dir, get_file_ids(rel_dir)[0])

        # system run
        sysrun_dir = os.path.join(DATA_DIR, trec_name, DICT_TREC_TYPE[trec_name][0])
        file_ids = get_file_ids(sysrun_dir)

        # get (r, ap, rp) for all topics and system runs
        dict_answer = defaultdict(dict)
        for run_id in file_ids:
            path_query_rt_file = os.path.join(sysrun_dir, run_id)
            eval_result = self.trec_eval(path_qrels_file=path_qrels_file, path_query_rt_file=path_query_rt_file)

            # per topic
            for topic_id in eval_result.queries.keys():
                r = eval_result.get_measure_score_by_query('num_rel', topic_id)
                t = eval_result.get_measure_score_by_query('num_ret', topic_id)

                ap = eval_result.get_measure_score_by_query('map', topic_id)
                rp = eval_result.get_measure_score_by_query('Rprec', topic_id)
                p30 = eval_result.get_measure_score_by_query('P_30', topic_id)
                dcg = 0  # stub
                rr = eval_result.get_measure_score_by_query('recip_rank', topic_id)
                bpref = eval_result.get_measure_score_by_query('bpref', topic_id)

                dict_answer[run_id.strip()][topic_id.strip()] = (r, r, t, ap, rp, p30, dcg, rr, 0, bpref)

            # all
            total_r = eval_result.get_total_measure_score('num_rel')
            total_t = eval_result.get_total_measure_score('num_ret')

            total_ap = eval_result.get_total_measure_score('map')
            total_rp = eval_result.get_total_measure_score('Rprec')
            total_p30 = eval_result.get_total_measure_score('P_30')
            total_dcg = 0  # stub

            total_rr = eval_result.get_total_measure_score('recip_rank')
            total_bpref = eval_result.get_total_measure_score('bpref')

            dict_answer[run_id.strip()]['all_topics'] = (total_r, total_r, total_t, total_ap, total_rp, total_p30, total_dcg,
                                                         total_rr, total_bpref)

        return dict_answer

    def get_answer(self, topic_id, list_sysrun_name):
        """Test. The result can be compared with that of Preprocess."""

        list_answer = []
        for run_id in list_sysrun_name:
            list_answer.append(self.dict_answer[run_id.strip()][topic_id.strip()])

        return list_answer

    def get_system_answer(self):
        """Test. The result can be compared with the published results in TREC overview papers."""

        print('system  (R, AP, RP) ')
        for key in self.dict_answer.keys():
            print('{}  {} '.format(key, self.dict_answer[key]['all_topics']))
        return


if __name__ == '__main__':
    pass
