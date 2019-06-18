import tensorflow as tf
from collections import defaultdict
from pprint import pprint
import json

bucket = "gs://redbert/final-models"
modes = ['masked', 'unmasked']
max_fold = 10
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

acc = {}

#step_to_check = 910
step_to_check = 1517

for mode in modes:
    acc[mode] = {}
    step_counts = defaultdict(int)

    for seed in seeds:

        for fold in range(1, max_fold+1):
            key = "seed{}-fold{}of{}".format(seed, fold, max_fold)
            acc[mode][key] = []

            path = "{}/{}/seed{}-fold{}of{}/eval/*".format(bucket, mode, seed, fold, max_fold)
            files = tf.gfile.Glob(path)

            for f in files:
                for e in tf.train.summary_iterator(f):
                    if (e.step == step_to_check):
                        for v in e.summary.value:
                            if v.tag == 'eval_accuracy':
                                acc[mode][key].append(v.simple_value)

                                #accuracies.get(mode, []).append(v.simple_value)
                                #print(accuracies)

pprint(acc)

with open("./results_second.json", "w") as f:
    json.dump(acc, f)
