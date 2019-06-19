from glob import glob
from scipy import stats
import pandas as pd
import json
import math

pairs = []

# Compare unmasked and masked performance, calculating T for each
pairs.append(("./results/evaluation/l2en_results.json",
    "./results/evaluation/l2enmask_results.json"))
pairs.append(("./results/evaluation/l1en_results.json",
    "./results/evaluation/l1enmask_results.json"))
pairs.append(("./results/evaluation/l2enfne_results.json",
    "./results/evaluation/l2enfnemask_results.json"))
pairs.append(("./results/evaluation/l1enfne_results.json",
    "./results/evaluation/l1enfnemask_results.json"))

for pair in pairs:
    a = pd.read_json(pair[0])
    b = pd.read_json(pair[1])
    t = stats.ttest_rel(a, b)

    print("Comparison of {}:".format(pair))
    print("Means: {}, {}".format(a.mean()[0], b.mean()[0]))
    print("The t-statistic is {} (unmasked=positive, masked=negative).".format(t.statistic))
    print("Two-tailed p-value of {}.".format(t.pvalue))
    print("***")

