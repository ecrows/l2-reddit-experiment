
#e.g. tf.gfile.Copy("gs://redbert/reddit-data/ground_truth_lines.txt", "/tmp/reddit-data/")
# !gsutil cp gs://redbert/reddit-data/l2_lines.txt /tmp/reddit-data
# !gsutil cp gs://redbert/reddit-data/l2_ne_lines.txt /tmp/reddit-data
# !gsutil cp gs://redbert/reddit-data/l1samplelines_masked.txt /tmp/reddit-data
# !gsutil cp gs://redbert/reddit-data/l1samplelines_topten_masked.txt /tmp/reddit-data
# !gsutil cp gs://redbert/reddit-data/l2_ne_lines.txt /tmp/reddit-data
# !gsutil cp gs://redbert/reddit-masked/*.txt /tmp/reddit-data
# !gsutil cp gs://redbert/reddit-data/l1english*.csv /tmp/reddit-data


l2_balanced = clip_and_relabel(l2df, len(gtdf), 0)

l2df = load_stripped_lines("/tmp/reddit-data/l2_lines.txt")
l2nedf = load_stripped_lines("/tmp/reddit-data/l2_ne_lines.txt")

mask_l2df = load_stripped_lines("/tmp/reddit-data/l2_lines_masked.txt")
mask_l2nedf = load_stripped_lines("/tmp/reddit-data/l2_ne_masked.txt")

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
