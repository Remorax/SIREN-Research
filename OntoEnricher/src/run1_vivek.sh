#!/bin/bash

#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

module load cuda/10.2
module load cudnn/7.6.5-cuda-10.2 

thresholds=(0.5 0.59 0.6 0.65 0.66)

for threshold in "${thresholds[@]}";
do
	python3 LSTM_optimized.py "data_use_"$threshold".pkl" "results_threshold_"$threshold".txt" "Output_threshold_"$threshold "threshold_"$threshold".pt"
done