# Copyright 2019 Google Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications by Evan Crothers, 2019 under Apache License, Version 2.0
# Requirements: Python3 and run 'pip install bert-tensorflow'

from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
#from google.colab import auth

import bertmodel
import sys

if len(sys.argv) < 2:
  print("This script requires an argument for masked or unmasked mode.")
  exit()

if sys.argv[1] == 'masked':
  MASK_MODE = 'masked'
elif sys.argv[1] == 'unmasked':
  MASK_MODE = 'unmasked'
else:
  print("Invalid mask mode.")
  exit()

#auth.authenticate_user()

timestamp = datetime.now().timestamp()
OUTPUT_DIR = 'validation-models/{}'.format(timestamp)
DO_DELETE = False
USE_BUCKET = True
BUCKET = 'redbert'

if USE_BUCKET:
  OUTPUT_DIR = 'gs://{}/{}'.format(BUCKET, OUTPUT_DIR)

if DO_DELETE:
  try:
    tf.gfile.DeleteRecursively(OUTPUT_DIR)
  except:
    # Doesn't matter if the directory didn't exist
    pass

tf.gfile.MakeDirs(OUTPUT_DIR)
print('***** Model output directory: {} *****'.format(OUTPUT_DIR))

seed = 44

# TODO: Convert to gfile calls
# !mkdir /tmp/reddit-data
# !gsutil cp gs://redbert/reddit-data/ground_truth_lines.txt /tmp/reddit-data
# !gsutil cp gs://redbert/reddit-data/reddit_random_lines.txt /tmp/reddit-data
# !gsutil cp gs://redbert/reddit-data/l2_lines.txt /tmp/reddit-data
# !gsutil cp gs://redbert/reddit-data/l2_ne_lines.txt /tmp/reddit-data
# !gsutil cp gs://redbert/reddit-data/l1samplelines_masked.txt /tmp/reddit-data
# !gsutil cp gs://redbert/reddit-data/l1samplelines_topten_masked.txt /tmp/reddit-data
# !gsutil cp gs://redbert/reddit-data/l2_ne_lines.txt /tmp/reddit-data
# !gsutil cp gs://redbert/reddit-masked/*.txt /tmp/reddit-data
# !gsutil cp gs://redbert/reddit-data/l1english*.csv /tmp/reddit-data

def load_stripped_lines(filename):
  with open(filename) as f:
    return pd.DataFrame([s.strip() for s in f.readlines()])

# TODO: The relabel part is generally unnecessary
def clip_and_relabel(data, length, label, seed=seed):
  balanced = data.sample(n=length, random_state=seed).copy()
  
  if len(balanced.columns) == 1:
    balanced.columns = ['sentence']
  elif len(balanced.columns) == 2:
    balanced.columns = ['sentence', 'label']
  
  balanced['label'] = label
  return balanced

gtdf = load_stripped_lines("/tmp/reddit-data/ground_truth_lines.txt")
l2df = load_stripped_lines("/tmp/reddit-data/l2_lines.txt")
rrdf = load_stripped_lines("/tmp/reddit-data/reddit_random_lines.txt")
l2nedf = load_stripped_lines("/tmp/reddit-data/l2_ne_lines.txt")

mask_gtdf = load_stripped_lines("/tmp/reddit-data/ground_truth_masked.txt")
mask_l2df = load_stripped_lines("/tmp/reddit-data/l2_lines_masked.txt")
mask_l2nedf = load_stripped_lines("/tmp/reddit-data/l2_ne_masked.txt")
mask_rrdf = load_stripped_lines("/tmp/reddit-data/reddit_random_masked.txt")
  
rr_balanced = clip_and_relabel(rrdf, len(gtdf), 0)
gt_balanced = clip_and_relabel(gtdf, len(gtdf), 1)
l2_balanced = clip_and_relabel(l2df, len(gtdf), 0)

rrmask_balanced = clip_and_relabel(mask_rrdf, len(gtdf), 0)
gtmask_balanced = clip_and_relabel(mask_gtdf, len(gtdf), 1)
l2mask_balanced = clip_and_relabel(mask_l2df, len(gtdf), 0)

l2ne_balanced = clip_and_relabel(l2nedf, len(gtdf), 0)
l2nemask_balanced = clip_and_relabel(mask_l2nedf, len(gtdf), 0)

# These two were generated offline in order to avoid reuploading too many GB
l1df = pd.read_csv("/tmp/reddit-data/l1english_sample.csv")
l1nedf = pd.read_csv("/tmp/reddit-data/l1english_topten_sample.csv")

mask_l1nedf = load_stripped_lines("/tmp/reddit-data/l1samplelines_topten_masked.txt")
mask_l1df = load_stripped_lines("/tmp/reddit-data/l1samplelines_masked.txt")

l1mask_balanced = clip_and_relabel(mask_l1df, len(gtdf), 0)
l1nemask_balanced = clip_and_relabel(mask_l1nedf, len(gtdf), 0)

"""Now that we have our data we can split it into test and train datasets."""
supported_modes = {
    'unmasked': rr_balanced.append(gt_balanced),
    'masked': rrmask_balanced.append(gtmask_balanced),
}

train, test = train_test_split(supported_modes[MASK_MODE], test_size=0.10)

print("Created training dataset for {}".format(MASK_MODE))

print("Train dataset breakdown:")
print(train.groupby('label').count())

print("Test dataset breakdown:")
print(test.groupby('label').count())

DATA_COLUMN = 'sentence'
LABEL_COLUMN = 'label'
label_list = [0, 1]

train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

tokenizer = bertmodel.create_tokenizer_from_hub_module()

MAX_SEQ_LENGTH = 128

# Convert our train and test features to InputFeatures that BERT understands.
train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

# Compute train and warmup steps from batch size
# These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
# Warmup is a period of time where hte learning rate 
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100

# Compute # train and warmup steps from batch size
num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

# Specify outpit directory and number of checkpoint steps to save
run_config = tf.estimator.RunConfig(
    model_dir=OUTPUT_DIR,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

model_fn = bertmodel.model_fn_builder(
  num_labels=len(label_list),
  learning_rate=LEARNING_RATE,
  num_train_steps=num_train_steps,
  num_warmup_steps=num_warmup_steps)

estimator = tf.estimator.Estimator(
  model_fn=bertmodel.model_fn,
  config=run_config,
  params={"batch_size": BATCH_SIZE})

"""Next we create an input builder function that takes our training feature set (`train_features`) and produces a generator. This is a pretty standard design pattern for working with Tensorflow Estimators"""

# Create an input function for training. drop_remainder = True for using TPUs.
train_input_fn = bert.run_classifier.input_fn_builder(
    features=train_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=True,
    drop_remainder=False)

print('Beginning Training!')
current_time = datetime.now()
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print("Training took time ", datetime.now() - current_time)

"""Evaluate model"""
test_input_fn = run_classifier.input_fn_builder(
    features=test_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)

estimator.evaluate(input_fn=test_input_fn, steps=None)

"""Evaluation against L2 datasets"""
l2_relabelled = l2_balanced
l2_relabelled['label'] = 0

l2mask_relabelled = l2mask_balanced
l2mask_relabelled['label'] = 0

l2ne_relabelled = l2ne_balanced
l2ne_relabelled['label'] = 0

l2nemask_relabelled = l2nemask_balanced
l2nemask_relabelled['label'] = 0

l1nemask_relabelled = l1nemask_balanced
l1nemask_relabelled['label'] = 0

l1mask_relabelled = l1mask_balanced
l1mask_relabelled['label'] = 0

l2mask_InputExamples = l2mask_relabelled.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

l2mask_features = bert.run_classifier.convert_examples_to_features(l2mask_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

l2_InputExamples = l2_relabelled.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

l2_features = bert.run_classifier.convert_examples_to_features(l2_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

l2nemask_InputExamples = l2nemask_relabelled.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

l2nemask_features = bert.run_classifier.convert_examples_to_features(l2nemask_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

l2ne_InputExamples = l2ne_relabelled.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

l2ne_features = bert.run_classifier.convert_examples_to_features(l2ne_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

l1_InputExamples = l1df.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

l1_features = bert.run_classifier.convert_examples_to_features(l1_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

l1ne_InputExamples = l1nedf.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

l1ne_features = bert.run_classifier.convert_examples_to_features(l1ne_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

l1mask_InputExamples = l1mask_relabelled.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

l1mask_features = bert.run_classifier.convert_examples_to_features(l1mask_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

l1nemask_InputExamples = l1nemask_relabelled.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

l1nemask_features = bert.run_classifier.convert_examples_to_features(l1nemask_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

# Evaluate misclassifications on random L2 comments.
l2_test_input_fn = run_classifier.input_fn_builder(
    features=l2_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)

l2mask_test_input_fn = run_classifier.input_fn_builder(
    features=l2mask_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)

l2ne_test_input_fn = run_classifier.input_fn_builder(
    features=l2ne_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)

l2nemask_test_input_fn = run_classifier.input_fn_builder(
    features=l2nemask_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)

# Evaluate misclassifications on random L1 English comments

l1_test_input_fn = run_classifier.input_fn_builder(
    features=l1_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)

l1ne_test_input_fn = run_classifier.input_fn_builder(
    features=l1ne_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)

l1mask_test_input_fn = run_classifier.input_fn_builder(
    features=l1mask_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)

l1nemask_test_input_fn = run_classifier.input_fn_builder(
    features=l1nemask_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)

# TODO: Output to file, save to cloud
estimator.evaluate(input_fn=l1_test_input_fn, steps=None)
estimator.evaluate(input_fn=l1ne_test_input_fn, steps=None)
estimator.evaluate(input_fn=l1mask_test_input_fn, steps=None)
estimator.evaluate(input_fn=l1nemask_test_input_fn, steps=None)
estimator.evaluate(input_fn=l2_test_input_fn, steps=None)
estimator.evaluate(input_fn=l2mask_test_input_fn, steps=None)
estimator.evaluate(input_fn=l2ne_test_input_fn, steps=None)
estimator.evaluate(input_fn=l2nemask_test_input_fn, steps=None)
