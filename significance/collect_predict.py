import tensorflow as tf
from collections import defaultdict
from pprint import pprint
from statistics import mean
import pandas as pd
import json
import sys
import os
import errno

bucket = "gs://redbert/final-models"
modes = ['masked', 'unmasked']
max_fold = 10
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

step_to_check = 1517

if len(sys.argv) < 2:
    print("Must specify task (e.g. l1en, l1enmask, l1enfne, l1enfnemask)")
    exit()

task = sys.argv[1]

fpr = []

if task in ["l1en", "l1enfne", "l2en", "l2enfne"]:
    mode = "unmasked"
elif task in ["l1enmask", "l1enfnemask", "l2enmask", "l2enfnemask"]:
    mode = "masked"
else:
    print("Unrecognized task.")
    exit()

for seed in seeds:
    for fold in range(1, max_fold+1):
        key = "seed{}-fold{}of{}".format(seed, fold, max_fold)
        path = "{}/{}/seed{}-fold{}of{}/test_results_{}.tsv".format(bucket, mode, seed, fold, max_fold, task)

        try:
            results = pd.read_csv(path, sep="\t", header=None, names=["0", "1"])

            total_len = results.shape[0]
            misclass = results.apply(lambda x: True if x["0"] < x["1"] else False, axis=1)
            error_count = len(misclass[misclass == True].index)

            fpr.append(error_count/total_len)
        except FileNotFoundError as e:
            print("No file at path {}. Continuing...".format(path))

print("Mean false-positive rate for {}: {}".format(task, mean(fpr)))

try:
    os.makedirs('results')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

with open("./results/{}_results.json".format(task), "w") as f:
    json.dump(fpr, f)
