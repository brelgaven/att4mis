#!/bin/bash
#SBATCH  --output=./log/train/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

source /itet-stor/gtombak/net_scratch/conda/etc/profile.d/conda.sh
conda activate pytcu10
python -u train_segmentation_source.py ./config/abide_caltech/abide_caltech_train_segmentation_cfg_01.py "$@"
