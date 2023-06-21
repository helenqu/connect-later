#!/bin/bash

num_procs=1
counter=0
infile_list=""

for entry in $1/plasticc_test_lightcurves*
do
    if ((counter < num_procs))
    then
        infile_list="$infile_list $entry"
        counter=$((counter+1))
    else
        echo $infile_list
        sbatch run_slurm_data_to_examples.sh "--infiles ${infile_list} --metadata_file $1/plasticc_test_metadata.csv.gz"
        infile_list="$entry"
        counter=1
    fi
done
# catch the stragglers
echo $infile_list
sbatch run_slurm_data_to_examples.sh "--infiles ${infile_list} --metadata_file $1/plasticc_test_metadata.csv.gz"
