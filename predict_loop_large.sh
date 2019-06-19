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
        echo 'Must specify a task (e.g. l1en, l1enfnemask)'
        exit 1
fi

TASKTYPE='none'
if [ $TASK = l1enmask ] || [ $TASK = l1enfnemask ] || [ $TASK = l2enfnemask ] || [ $TASK = l2enmask ]
then
  TASKTYPE='masked'
fi

if [ $TASK = l1en ] || [ $TASK = l1enfne ] || [ $TASK = l2en ] || [ $TASK = l2enfne ]
then
  TASKTYPE='unmasked'
fi

if [ $TASKTYPE = none ]
then
	echo 'Invalid task type!'
	exit 1
fi


export BERT_DIR=gs://redbert/bert-large-wwm
export DATA_DIR=gs://redbert/reddit-data

FOLDS=10
MAX_SEQ=$((FOLDS-1))

for FOLD_INDEX in $(seq 0 $MAX_SEQ); do
  MODEL_DIR=gs://redbert/final-models/${TASKTYPE}/seed$SEED-fold$((FOLD_INDEX+1))of$FOLDS
  python3 run_classifier.py \
    --task_name=$TASK \
    --do_train=false \
    --do_eval=false \
    --do_predict=true \
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

