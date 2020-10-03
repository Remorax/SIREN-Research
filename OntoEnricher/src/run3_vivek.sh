#!/bin/bash

#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

module load cuda/10.2
module load cudnn/7.6.5-cuda-10.2 

dropouts=(0.35 0.4 0.45 0.5 0.6 0.7 0.8)

for dropout in "${dropouts[@]}";
do
	python3 LSTM_output_dropout.py "data_use_0.86.pkl" "results_output_dropout_"$dropout".txt" "Output_output_dropout_"$dropout "output_dropout_"$dropout".pt" $dropout
done