#!/bin/bash

#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

module load cuda/10.2
module load cudnn/7.6.5-cuda-10.2 

dropouts=(0 0.05 0.1 0.15 0.2 0.25 0.3)

for dropout in "${dropouts[@]}";
do
	python3 LSTM_lstm_dropout.py "data_use_0.86.pkl" "results_lstm_dropout_"$dropout".txt" "Output_lstm_dropout_"$dropout "lstm_dropout_"$dropout".pt" $dropout
done