#coding=utf-8

import getpass
if 'danli' == getpass.getuser():
    PROJECT_HOME = '/Users/danli/Documents/Project/acsp'
elif 'dli1' == getpass.getuser():
    PROJECT_HOME = '/zfs/ilps-plex1/slurm/datastore/dli1/project/acsp'
else:
    PROJECT_HOME = ''  # add new repository here

DATA_DIR = PROJECT_HOME + '/data/Archive'
ORDER_DATA_DIR = PROJECT_HOME + '/data/order_archive'
PICKLE_DIR = PROJECT_HOME + '/data/pickle'
RESULT_DIR = PROJECT_HOME + '/data/sample/'
MIDWARE_DIR = PROJECT_HOME + '/data/midware'
EXP_DIR = PROJECT_HOME + '/data/eval'
TREC_EVAL_EXCUTE = PROJECT_HOME + '/code/trec_eval.9.0/trec_eval'


RELEVANT = 1
NON_RELEVANT = 0

JUDGED = 1
UNJUDGED = 0

SAMPLING_TYPE_ARGMAX = 1
SAMPLING_TYPE_SAMPLE_WOR = 2
SAMPLING_TYPE_SAMPLE_WR = 3

WEIGHT_TYPE_RANK = 0b0010
WEIGHT_TYPE_RANK_DIRICHLET = 0b0011
WEIGHT_TYPE_AP = 0b0100
WEIGHT_TYPE_AP_DIRICHLET = 0b0101
WEIGHT_TYPE_BPREF = 0b1000
WEIGHT_TYPE_BPREF_DIRICHLET = 0b1001
WEIGHT_TYPE_ESAP = 0b10000
WEIGHT_TYPE_ESAP_DIRICHLET = 0b10001

TEST_COLLECTION_DEPTH = 100

STR_NULL_DOC = 'NULL_DOC'

BATCH_NUM = 3

CUTOFF_30 = 30

TRAIN_SIZE = 0.8

PERCENTAGES = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.20]

DICT_TREC_TOPIC = {
    'TREC-5': range(251, 301),
    'TREC-6': range(301, 351),
    'TREC-7': range(351, 401),
    'TREC-8': range(401, 451),
    'TREC-9': range(451, 501),
    'TREC-10': range(501, 551),
    'TREC-11': range(551, 601),
}

DICT_TREC_TYPE = {
    #          (run dir, qrel dir, data type)
    'TREC-5': ('Adhoc/Runs/CategoryA', 'Adhoc/QRels', 'Adhoc'),
    'TREC-6': ('Adhoc/Runs/CategoryA', 'Adhoc/QRels', 'Adhoc'),
    'TREC-7': ('Adhoc/Runs', 'Adhoc/QRels', 'Adhoc'),
    'TREC-8': ('Adhoc/Runs', 'Adhoc/QRels', 'Adhoc'),
    'TREC-9': ('Web/Runs', 'Web/QRels', 'Web'),
    'TREC-10': ('Web/Runs', 'Web/QRels', 'Web'),
    'TREC-11': ('Web/Runs', 'Web/QRels', 'Web'),
}

LIST_RUNS = [('TREC-5', 61),
             ('TREC-6', 74),
             ('TREC-7', 103),
             ('TREC-8', 129),
             ('TREC-9', 104),
             ('TREC-10', 97),
             ('TREC-11', 69)]

LIST_GROUPS = [('TREC-5', 21),
               ('TREC-6', 29),
               ('TREC-7', 42),
               ('TREC-8', 41),
               ('TREC-9', 23),
               ('TREC-10', 29),
               ('TREC-11', 16)]
