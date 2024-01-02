#!/bin/bash

num_procs=10
counter=0
infile_list=""

for entry in $1/*.h5
do
    if ((counter < num_procs))
    then
        infile_list="$infile_list $entry"
        counter=$((counter+1))
    else
        echo $infile_list
        # echo "--outdir ${2}"
        is_sdss=$(echo $entry | grep -c "sdss")
        # add --sdss flag if the input file is from SDSS
        if ((is_sdss > 0))
        then
            echo "SDSS"
            sbatch run_slurm_data_to_examples.sh "--infiles ${infile_list} --outdir ${2} --sdss"
        else
            echo "NOT SDSS"
            sbatch run_slurm_data_to_examples.sh "--infiles ${infile_list} --outdir ${2}"
        fi
        infile_list="$entry"
        counter=1
    fi
done
# catch the stragglers
echo $infile_list
is_sdss=$(echo $infile_list | grep -c "sdss")
# add --sdss flag if the input file is from SDSS
if ((is_sdss > 0))
then
    echo "SDSS"
    sbatch run_slurm_data_to_examples.sh "--infiles ${infile_list} --outdir ${2} --sdss"
else
    echo "NOT SDSS"
    sbatch run_slurm_data_to_examples.sh "--infiles ${infile_list} --outdir ${2}"
fi
