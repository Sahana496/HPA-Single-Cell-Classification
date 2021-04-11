#!/bin/bash

#SBATCH --job-name=maskrcnn
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

module purge


unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

if [[ $(hostname -s) =~ ^g ]]; then nv="--nv"; fi

singularity exec $nv \
	    --overlay /scratch/su575/DLM/pytorch1.7.0-cuda11.0.ext3:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "
	    source /ext3/env.sh
	    python mask-rcnn.py
	    "





