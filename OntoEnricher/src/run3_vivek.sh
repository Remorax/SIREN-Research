#!/bin/bash

#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

module load cuda/10.2
module load cudnn/7.6.5-cuda-10.2 

emb_dropouts=(0.35)
hidden_dropouts=(0.8)
output_dropouts=(0)
NUM_LAYERS=(2)
# HIDDEN_DIMS=((120, 0), (120, 60), (120, 90), (60, 0), (180, 0), (180, 120), (180, 60), (250, 0), (250, 120), (250, 60), (500, 0), (500, 250), (500, 150), (500, 60), (750, 300), (750, 150), (750, 500))
# HIDDEN_DIMS=((500, 300), (500, 250), (500, 180), (500, 120), (500, 90), (500, 60), (300, 180), (300, 120), (300, 90), (300, 60), (250, 120), (250, 60), (250, 180), (250, 90), (180, 90), (180, 120), (180, 60), (120, 90), (120, 60), (90, 60))

for emb_dropout in "${emb_dropouts[@]}";
do
	for hidden_dropout in "${hidden_dropouts[@]}";
	do
		for output_dropout in "${output_dropouts[@]}";
		do
			for NUM_LAYER in "${NUM_LAYERS[@]}";
			do
				for i in "500 90" "500 60"
				do
					set -- $i
					echo $1 and $2
					python3 test_instance.py data_instances_security.pkl "results_security"$emb_dropout"_"$hidden_dropout"_"$output_dropout"_"$NUM_LAYER"_"$1"_"$2".txt" "Output_security"$emb_dropout"_"$hidden_dropout"_"$output_dropout"_"$NUM_LAYER"_"$1"_"$2 "security"$emb_dropout"_"$hidden_dropout"_"$output_dropout"_"$NUM_LAYER"_"$1"_"$2".pt" $emb_dropout $hidden_dropout $output_dropout $NUM_LAYER $1 $2
				done	
			done
		done
	done
done