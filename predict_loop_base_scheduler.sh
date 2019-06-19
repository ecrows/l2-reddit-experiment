#!/bin/bash

WORKER=$1
if [ -z "$WORKER" ]
then
        echo 'Must specify a TPU'
        exit 1
fi

TASK=$2
if [ -z "$TASK" ]
then
        echo 'Must specify a task (e.g. l2enfnemask)'
        exit 1
fi

for ((i = 1 ; i <= 10; i++)); do
  echo "Running predict loop for seed $i"
  /bin/bash predict_loop_base.sh $WORKER $i $TASK
done

