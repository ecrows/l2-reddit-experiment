import tensorflow as tf
from collections import defaultdict
from pprint import pprint
import json
import os
import errno
import sys

if len(sys.argv) < 2:
    print("Require bucket folder (e.g. base-models, final-models)")
    exit()

if len(sys.argv) < 3:
    print("Require step to get (e.g. 1517, 910)")
    exit()

bucket = "gs://redbert/{}".format(sys.argv[1])
modes = ['masked', 'unmasked']
max_fold = 10
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

step_to_check = int(sys.argv[2])

to_retrieve = ['eval_accuracy', 'auc', 'f1_score', 'precision', 'recall']
stats = {}

for r in to_retrieve:
    stats[r] = {}
    for mode in modes:
        stats[r][mode] = {}


def get_field_from_files(files, field):
    for f in files:
        for e in tf.train.summary_iterator(f):
            if (e.step == step_to_check):
                for v in e.summary.value:
                    if v.tag == field:
                        return v.simple_value
    return None


for mode in modes:
    for seed in seeds:
        for fold in range(1, max_fold+1):
            key = "seed{}-fold{}of{}".format(seed, fold, max_fold)
            path = "{}/{}/seed{}-fold{}of{}/eval/*".format(bucket, mode, seed, fold, max_fold)
            files = tf.gfile.Glob(path)

            for r in to_retrieve:
                stats[r][mode][key] = get_field_from_files(files, r)

try:
    os.makedirs('results')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.makedirs('results/classifier')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


for r in to_retrieve:
    with open("./results/classifier/{}_results.json".format(r), "w") as f:
        json.dump(stats[r], f)
