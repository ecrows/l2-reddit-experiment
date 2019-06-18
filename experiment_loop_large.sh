#!/bin/bash

# Start a TPU
TPU_NAME=$1
if [ -z "$TPU_NAME" ]
then
        echo 'Must specify a name for the TPU'
        exit 1
fi

SEED=$2
if [ -z "$SEED" ]
then
        echo 'Must specify a seed'
        exit 1
fi

TASK=$3
if [ -z "$TASK" ]
then
        echo 'Must specify a task (e.g. RRGT, RRGTmask)'
        exit 1
fi

TASKTYPE='none'
if [ $TASK = RRGT ]
then
  TASKTYPE='unmasked'
fi

if [ $TASK = RRGTmask ]
then
  TASKTYPE='masked'
fi


export BERT_DIR=gs://redbert/bert-large-wwm
export DATA_DIR=gs://redbert/reddit-data

FOLDS=10
MAX_SEQ=$((FOLDS-1))

for FOLD_INDEX in $(seq 0 $MAX_SEQ); do
  MODEL_DIR=gs://redbert/final-models/${TASKTYPE}/seed$SEED-fold$((FOLD_INDEX+1))of$FOLDS
  python3 run_classifier.py \
    --task_name=$TASK \
    --do_train=true \
    --do_eval=true \
    --do_predict=false \
    --data_dir=$DATA_DIR \
    --vocab_file=$BERT_DIR/vocab.txt \
    --bert_config_file=$BERT_DIR/bert_config.json \
    --init_checkpoint=$BERT_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=64 \
    --learning_rate=2e-5 \
    --num_train_epochs=5.0 \
    --folds=$FOLDS \
    --fold_index=$FOLD_INDEX \
    --seed=$SEED \
    --use_tpu=True \
    --tpu_name=$TPU_NAME \
    --output_dir=$MODEL_DIR
done

