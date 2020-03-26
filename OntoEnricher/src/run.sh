#!/usr/bin/env bash

corpus=$1 
folder=$2
prefix=$3

n=`grep "" -c $corpus | awk '{ print $1 }'`
m=5
n=$((($n + 1)/$m))

declare -a path_thresholds=(3 7 10 15 20 25 50)
parts=( $(seq 1 $m ) )

gsplit -l $n $corpus $corpus"_split_" --numeric-suffixes=1;

for x in "${parts[@]}"
do
	corpus_part=$corpus"_split_"$x
	( python3 corpus_parser.py $corpus_part ) &
done
wait


parsed_final=$corpus"_parsed"
cat $corpus"_split_"*"_parsed" > $parsed_final


for x in "${parts[@]}"
do
	parsed_final_part=$corpus"_split_"$x"_parsed"
	( awk -F "\t" '{relations[$3]++} END{for(relation in relations){print relation"\t"relations[relation]}}' $parsed_final_part > $corpus"_paths_"$x ) &
done
wait
echo "Done till here.. part 3"

paths=$folder"all_paths"
cat $corpus"_paths_"* > $paths
# rm $corpus"_paths_"*

# declare -a path_thresholds=(3 7 10 15 20 25 50)
declare -a path_thresholds=(3 5)
for n in "${path_thresholds[@]}"
do
	mkdir $folder$prefix"_threshold_"$n
	( awk -F "\t" '{i[$1]+=$2} END{for(x in i){ if (i[x] >= '$n') print x } }' $paths > $folder$prefix"_threshold_"$n'/filtered_paths'  ) &
done
wait
# rm $paths

awk -F$'\t' '{if (a[$1] == 0) {a[$1] = -1; print $1}}' $parsed_final > $output"xterms" & PIDLEFT=$!
awk -F$'\t' '{if (a[$2] == 0) {a[$2] = -1; print $2}}' $parsed_final > $output"yterms" & PIDRIGHT=$!

wait $PIDLEFT
wait $PIDRIGHT
cat $output"xterms" $output"yterms" | sort -u > $output"terms";
# rm $output"xterms" $output"yterms" $parsed_final

echo 'Creating the resource from the triplets file...'
for n in "${path_thresholds[@]}"
do
	paths_folder=$folder$prefix"_threshold_"$n
	( python3 path_terms_indexer.py $paths_folder $output"terms" $prefix 1; ) &
done
wait
for n in "${path_thresholds[@]}"
do
	paths_folder=$folder$prefix"_threshold_"$n
	for x in "${parts[@]}"
	do
		parsed_final_part=$corpus"_split_"$x"_parsed"
		( python3 path_terms_indexer.py $paths_folder $parsed_final_part $prefix 2; ) &
	done
	wait

	for x in "${parts[@]}"
	do
		triplet_part_file=$paths_folder"/triplet_id_"$x
		triplet_count_file=$paths_folder"/triplet_count_"$x
		( awk -F "\t" '{relations[$0]++} END{for(relation in relations){print relation"\t"relations[relation]}}' $triplet_part_file > $triplet_count_file ) &
	done
	wait

	cat $paths_folder"/triplet_count_"* > $paths_folder"/triplet_count";

	gawk -F $'\t' '{ matrix[$1][$2][$3]+=$4; } END{for (x in matrix) {for (y in matrix[x]) {for (path in matrix[x][y]) {print x, y, path, matrix[x][y][path]}}}}' $paths_folder"/triplet_count" > $paths_folder"/final_count"

	# python3 path_terms_indexer.py $paths_folder $paths_folder"_final_count" $prefix 3;
done




# gawk -F $'\t' '{ matrix[$1][$2][$3]+=$4; } END{for (x in matrix) {for (y in matrix[x]) {for (path in matrix[x][y]) {print x, y, path, matrix[x][y][path]}}}}' $paths_folder"/triplet_count" > $paths_folder"/final_count"

# # ls
# rm id_triplet_file_temp id_triplet_file_* $triplet_file"_"*;
# for n in "${path_thresholds[@]}"
# do
# 	paths_folder=$folder$prefix"_threshold_"$n
	
# done

# # You can delete triplet_file now and keep only id_triplet_file which is more efficient, or delete both.

# rm $corpus"_split_"*