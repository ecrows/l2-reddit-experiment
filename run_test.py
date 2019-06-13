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
from bert import modeling
from bert import tokenization
from tensorflow.errors import AlreadyExistsError
import tensorflow.contrib.tpu
import tensorflow.contrib.cluster_resolver

#import bertmodel
import sys

#import tempfile
import subprocess

BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 1.0
WARMUP_PROPORTION = 0.1
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100

DATA_COLUMN = 'sentence'
LABEL_COLUMN = 'label'
label_list = [0, 1]

MAX_SEQ_LENGTH = 128

gtdf = load_stripped_lines("/tmp/reddit-data/ground_truth_lines.txt")
rrdf = load_stripped_lines("/tmp/reddit-data/reddit_random_lines.txt")

rr_balanced = clip_and_relabel(rrdf, len(gtdf), 0)
gt_balanced = clip_and_relabel(gtdf, len(gtdf), 1)

rr_balanced.append(gt_balanced)

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=seed)
X = supported_modes[MASK_MODE]

for train_index, test_index in kf.split(X):
    break

train, test = X.iloc[train_index], X.iloc[test_index]

test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

model_fn = bert.run_classifier.model_fn_builder(
  bert_config=bert_config,
  num_labels=len(label_list),
  init_checkpoint=init_checkpoint,
  learning_rate=LEARNING_RATE,
  num_train_steps=num_train_steps,
  num_warmup_steps=num_warmup_steps,
  use_tpu=True,
  use_one_hot_embeddings=True) # Last one because TPU true?

#model_fn = bertmodel.model_fn_builder(
  #num_labels=len(label_list),
  #learning_rate=LEARNING_RATE,
  #num_train_steps=num_train_steps,
  #num_warmup_steps=num_warmup_steps)

estimator = tf.contrib.tpu.TPUEstimator(
  model_fn=model_fn,
  use_tpu=FLAGS.use_tpu,
  train_batch_size=BATCH_SIZE,
  eval_batch_size=BATCH_SIZE,
  predict_batch_size=BATCH_SIZE,
  config=run_config
  )

# Create an input function for training. drop_remainder = True for using TPUs.
train_input_fn = bert.run_classifier.input_fn_builder(
    features=train_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=True,
    drop_remainder=True
)

eval_steps = int(len(test_features) // BATCH_SIZE)

"""Evaluate model"""
test_input_fn = bert.run_classifier.input_fn_builder(
    features=test_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=True
)

results = estimator.evaluate(input_fn=test_input_fn, steps=eval_steps)
