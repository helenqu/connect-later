#!/bin/bash

num_procs=32
counter=0
infile_list=""

for entry in $1/prepr*.csv
do
    if ((counter < num_procs))
    then
        infile_list="$infile_list $entry"
        counter=$((counter+1))
    else
        echo $infile_list
        sbatch run_slurm_data_to_examples.sh "--infiles ${infile_list}"
        infile_list="$entry"
        counter=1
    fi
done
