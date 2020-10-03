#!/bin/bash

#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

module load cuda/10.2
module load cudnn/7.6.5-cuda-10.2 

thresholds=(0.92 0.93 0.94 0.95 0.96)

for threshold in "${thresholds[@]}";
do
	python3 test.py "data_use_"$threshold".pkl" "results_threshold_"$threshold".txt" "Output_threshold_"$threshold "threshold_"$threshold".pt"
done
