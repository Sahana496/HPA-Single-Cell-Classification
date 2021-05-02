#!/bin/bash

#SBATCH --job-name=maskrcnn1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

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
	    python -u mask-rcnn.py --pretrained --batch-size=16 --checkpoint-dir checkpoints/resnet101 --epochs=15 --model=resnet101
	    "





