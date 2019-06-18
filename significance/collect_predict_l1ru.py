import tensorflow as tf
from collections import defaultdict
from pprint import pprint
from statistics import mean
import pandas as pd
import json
import os

bucket = "gs://redbert/final-models/"
modes = ['masked', 'unmasked']
max_fold = 10
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

step_to_check = 1517

task = "l1enfnemask"
stats = {}

fpr = []


for mode in modes:
    for seed in seeds:
        for fold in range(1, max_fold+1):
            key = "seed{}-fold{}of{}".format(seed, fold, max_fold)
            path = "{}/{}/seed{}-fold{}of{}/test_results_{}.tsv".format(bucket, mode, seed, fold, max_fold, task)

            results = pd.read_csv(path, sep="\t", header=None, names=["0", "1"])

            total_len = results.shape[0]
            misclass = results.apply(lambda x: True if x["0"] < x["1"] else False, axis=1)
            error_count = len(misclass[misclass == True].index)

            fpr.append(error_count/total_len)

print("Mean false-positive rate for {}: {}".format(task, mean(fpr)))

try:
    os.makedirs('results')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

with open("./results/{}_results.json".format(task), "w") as f:
    json.dump(fpr, f)