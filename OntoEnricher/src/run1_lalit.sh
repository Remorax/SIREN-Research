#!/bin/bash

#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

module load cuda/10.2
module load cudnn/7.6.5-cuda-10.2 

emb_dropouts=(0.3 0.35)
hidden_dropouts=(0)
output_dropouts=(0)

for emb_dropout in "${emb_dropouts[@]}";
do
	for hidden_dropout in "${hidden_dropouts[@]}";
	do
		for output_dropout in "${output_dropouts[@]}";
		do
			python3 LSTM_dropout.py data_use_0.86.pkl "results_dropout_"$emb_dropout"_"$hidden_dropout"_"$output_dropout"_.txt" "Output_dropout_"$emb_dropout"_"$hidden_dropout"_"$output_dropout "dropout_"$emb_dropout"_"$hidden_dropout"_"$output_dropout".pt" $emb_dropout $hidden_dropout $output_dropout
		done
	done
done