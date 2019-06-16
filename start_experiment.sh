export BERT_BASE_DIR=gs://redbert/bert-base
export DATA_DIR=gs://redbert/reddit-data
export MODEL_DIR=gs://redbert/mrpc_model

python3 run_classifier.py \
  --task_name=RRGT \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=64 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --folds=10 \
  --fold_index=0 \
  --seed=44 \
  --use_tpu=True \
  --tpu_name=$TPU_NAME \
  --output_dir=$MODEL_DIR
