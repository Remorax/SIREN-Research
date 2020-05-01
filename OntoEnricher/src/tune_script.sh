
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