#!/bin/bash

for ((i = 1 ; i <= 10 ; i++)); do
  echo "Running predict loop for seed $i"
  /bin/bash predict_loop_large.sh worker4 $i l2enmask
done

