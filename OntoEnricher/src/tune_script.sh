

#!/bin/bash

#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=10240
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

module load cuda/10.2
module load cudnn/7.6.5-cuda-10.2 


declare -a hidden_dim=(60 90 120) # Different frequencies considered while creating frequent paths
declare -a layers=(2 4 6) # Maximum path lengths considered while extracting paths
declare -a dropouts=(0.3 0.5 0.8) # Maximum path lengths considered while extracting paths

for hidden in "${hidden_dim[@]}"
do
	for layer in "${layers[@]}"
	do
		for dropout in dropouts:
		do
			(python3 tune_params.py hidden layer dropout) &
		done
	done
done
wait
