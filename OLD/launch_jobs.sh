#!/bin/bash

date
for lr in 0.001 0.00001
do
    sbatch perlmutter_job.sh $lr
done

