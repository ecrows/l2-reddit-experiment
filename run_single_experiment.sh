export TPU_NAME=worker1

SEED=44
FOLD_INDEX=0
FOLDS=10

export BERT_BASE_DIR=gs://redbert/bert-base
export DATA_DIR=gs://redbert/reddit-data
export MODEL_DIR=gs://redbert/validation-models/againtest2-seed$SEED-fold$((FOLD_INDEX+1))of$FOLDS

python3 run_classifier.py \
  --task_name=RRGT \
  --do_train=true \
  --do_eval=true \
  --do_predict=false \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=64 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --folds=$FOLDS \
  --fold_index=$FOLD_INDEX \
  --seed=$SEED \
  --use_tpu=True \
  --tpu_name=$TPU_NAME \
  --output_dir=$MODEL_DIR
