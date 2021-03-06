from glob import glob
from scipy import stats
import numpy as np
import pandas as pd
import json
import math

a = pd.read_json("./results/classifier/eval_accuracy_results.json")

masked_seeds = a['masked']
unmasked_seeds = a['unmasked']

print("Collected {} masked samples and {} unmasked samples.".format(masked_seeds.shape[0], unmasked_seeds.shape[0]))

x = unmasked_seeds - masked_seeds

k = 10 # number of folds
r = 10 # number of experiments
kr = k*r # total number of runs
test_train_ratio = 1/9 # ratio of test examples to train examples

numerator = (1/kr) * x.sum()
denominator = math.sqrt(((1/kr) + test_train_ratio) * x.var())

t = numerator / denominator
pval = stats.t.sf(np.abs(t), kr-1)*2 # two-sided p-value

print("The t-statistic is {} (unmasked=positive, masked=negative).".format(t))
print("Two-sided p-value of {}".format(pval))

rfiles = glob("./results/classifier/*.json")

# Calculate means
for f in rfiles:
    r = pd.read_json(f)
    masked_vals = r['masked']
    unmasked_vals = r['unmasked']
    print(f)
    print("Masked mean: {:.4f}".format(masked_vals.mean()))
    print("Unmasked mean: {:.4f}".format(unmasked_vals.mean()))
