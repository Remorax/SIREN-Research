#!/bin/bash

#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=10240
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

module load cuda/10.2
module load cudnn/7.6.5-cuda-10.2 

rm -rf Logs

files=($(ls -d ../junk/wiki2vec_resolved_*))
for current in "${files[@]}";
do
	echo "Doing "$current
	python3 main_input.py $current
done
