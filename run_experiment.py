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

from sklearn.model_selection import KFold
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import os
import errno
from datetime import datetime
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from tensorflow.errors import AlreadyExistsError

import bertmodel
import sys

if len(sys.argv) < 3:
  print("This script requires an argument for masked or unmasked mode, as well as for random seed.")
  exit()

if sys.argv[1] == 'masked':
  MASK_MODE = 'masked'
elif sys.argv[1] == 'unmasked':
  MASK_MODE = 'unmasked'
else:
  print("Invalid mask mode.")
  exit()

# TODO: Some sort of validation probably. Use a library, seriously man.
seed = int(sys.argv[2])

#auth.authenticate_user()

timestamp = int(datetime.now().timestamp()*1000)
NUM_FOLDS = 10
OUTPUT_DIR = 'validation-models/seed{}_{}'.format(seed, timestamp)
#DO_DELETE = False
USE_BUCKET = True
BUCKET = 'redbert'

if USE_BUCKET:
  OUTPUT_DIR = 'gs://{}/{}'.format(BUCKET, OUTPUT_DIR)

#if DO_DELETE:
#  try:
#    tf.gfile.DeleteRecursively(OUTPUT_DIR)
#  except:
    # Doesn't matter if the directory didn't exist
#    pass


try:
    os.mkdir("/tmp/reddit-data")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

files_to_get = [
        "gs://redbert/reddit-data/ground_truth_lines.txt",
        "gs://redbert/reddit-data/ground_truth_masked.txt",
        "gs://redbert/reddit-data/reddit_random_lines.txt",
        "gs://redbert/reddit-data/reddit_random_masked.txt"
        ]

for file in files_to_get:
    try:
        basename = file[file.rindex("/")+1:]
        tf.gfile.Copy(file, "/tmp/reddit-data/{}".format(basename))
    except AlreadyExistsError as e:
        # Fine if we've already downloaded it, just continue
        print("File {} already exists. Skipping download...".format(file))
        pass

    #info = tf.gfile.Stat(file)
    #print(info)

    # TODO: Avoid doing it twice
    #if tf.gfile.Exists("/tmp/reddit-data/(basename)

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
rrdf = load_stripped_lines("/tmp/reddit-data/reddit_random_lines.txt")

mask_gtdf = load_stripped_lines("/tmp/reddit-data/ground_truth_masked.txt")
mask_rrdf = load_stripped_lines("/tmp/reddit-data/reddit_random_masked.txt")
  
rr_balanced = clip_and_relabel(rrdf, len(gtdf), 0)
gt_balanced = clip_and_relabel(gtdf, len(gtdf), 1)

rrmask_balanced = clip_and_relabel(mask_rrdf, len(gtdf), 0)
gtmask_balanced = clip_and_relabel(mask_gtdf, len(gtdf), 1)

"""Now that we have our data we can split it into test and train datasets."""
supported_modes = {
    'unmasked': rr_balanced.append(gt_balanced),
    'masked': rrmask_balanced.append(gtmask_balanced)
}

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=seed)

X = supported_modes[MASK_MODE]

fold = 0

for train_index, test_index in kf.split(X):
    fold += 1
    model_path = "{}_fold{}of{}".format(OUTPUT_DIR, fold, NUM_FOLDS)
    tf.gfile.MakeDirs(model_path)
    print('***** Model output directory: {} *****'.format(model_path))

    print("TRAIN:", train_index, "TEST:", test_index)
    train, test = X.iloc[train_index], X.iloc[test_index]

    print("Created training dataset for {}".format(MASK_MODE))

    print("Train dataset breakdown:")
    print(train.groupby('label').count())

    print("Test dataset breakdown:")
    print(test.groupby('label').count())

    DATA_COLUMN = 'sentence'
    LABEL_COLUMN = 'label'
    label_list = [0, 1]

    train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None,
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
        model_dir=model_path,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

    model_fn = bertmodel.model_fn_builder(
      num_labels=len(label_list),
      learning_rate=LEARNING_RATE,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(
      model_fn=model_fn,
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

    results = estimator.evaluate(input_fn=test_input_fn, steps=None)

    print("Printing results...")
    print(results)

