#!/usr/bin/env bash

corpus=$1 # The text corpus to process
folder=$2 # Folder which contains all files created during script (will be auto-deleted)
prefix=$3 # DB File prefix

echo "Corpus: "$corpus", Folder: "$folder", DB File prefix: "$prefix

n=`grep "" -c $corpus | awk '{ print $1 }'`

m=5
n=$((($n + 1)/$m))

echo "Splitting corpus into "$m" parts" 

declare -a path_thresholds=(3 7) # Different frequencies considered while creating frequent paths
declare -a maxlens=(4 6) # Maximum path lengths considered while extracting paths

parts=( $(seq 1 $m ) )

echo -e "\n\nTunable Parameters:\n\nPath Frequencies: "${path_thresholds[*]}"\nMax Lengths of paths: " ${maxlens[*]}"\n\n"

echo "Stage 1/3 : Splitting corpus..."

gsplit -l $n $corpus $corpus"_split_" --numeric-suffixes=1;

echo "Stage 2/3 : Parsing corpus..."

for x in "${parts[@]}"
do
	corpus_part=$corpus"_split_"$x
	( python3 corpus_parser.py $corpus_part ) &
done
wait

echo "Stage 3/3: The main stage: tuning of parameters..."
index=0
for maxlen in "${maxlens[@]}"
do
	((index++))
	
	echo -e "\t"$((index*100/${#maxlens[@]}))"% done: Maximum Path Length: "$maxlen
	parsed_final=$corpus"_"$maxlen"_parsed"
	cat $corpus"_split_"*"_"$maxlen"_parsed" > $parsed_final

	echo -e "\tStep: Counting relations..."
	for x in "${parts[@]}"
	do
		parsed_final_part=$corpus"_split_"$x"_"$maxlen"_parsed"
		( awk -F "\t" '{relations[$3]++} END{for(relation in relations){print relation"\t"relations[relation]}}' $parsed_final_part > $corpus"_paths_"$x"_"$maxlen ) &
	done
	wait
	

	paths=$folder"all_paths_"$maxlen
	cat $corpus"_paths_"*"_"$maxlen > $paths
	rm $corpus"_paths_"*"_"$maxlen

	echo -e "\tStep: Filtering common paths..."
	for n in "${path_thresholds[@]}"
	do
		echo -e "\t\tPath threshold: "$n
		mkdir $folder$prefix"_threshold_"$n"_"$maxlen
		( awk -F "\t" '{i[$1]+=$2} END{for(x in i){ if (i[x] >= '$n') print x } }' $paths > $folder$prefix"_threshold_"$n"_"$maxlen'/filtered_paths'  ) &
	done
	wait
	rm $paths

	echo -e "\tStep: Creating word files..."
	awk -F$'\t' '{if (a[$1] == 0) {a[$1] = -1; print $1}}' $parsed_final > $folder"xterms_"$maxlen & PIDLEFT=$!
	awk -F$'\t' '{if (a[$2] == 0) {a[$2] = -1; print $2}}' $parsed_final > $folder"yterms_"$maxlen & PIDRIGHT=$!

	wait $PIDLEFT
	wait $PIDRIGHT
	cat $folder"xterms_"$maxlen $folder"yterms_"$maxlen | sort -u > $folder"terms_"$maxlen;
	rm $folder"xterms_"$maxlen $folder"yterms_"$maxlen $parsed_final

	echo -e "\tStep: Creating term and path db files..."
	for n in "${path_thresholds[@]}"
	do
		paths_folder=$folder$prefix"_threshold_"$n"_"$maxlen
		( python3 path_terms_indexer.py $paths_folder $folder"terms_"$maxlen $prefix 1; ) &
	done
	wait

	rm $folder"terms_"$maxlen 
	rm $folder$prefix"_threshold_"*"_"$maxlen"/"filtered_paths

	echo -e "\tStep: Processing triplets..."
	for n in "${path_thresholds[@]}"
	do
		echo -e "\t\tPath threshold: "$n

		paths_folder=$folder$prefix"_threshold_"$n"_"$maxlen

		# Creating an ID file for the parsed triplets
		for x in "${parts[@]}"
		do
			parsed_final_part=$corpus"_split_"$x"_"$maxlen"_parsed"
			( python3 path_terms_indexer.py $paths_folder $parsed_final_part $prefix 2; ) &
		done
		wait

		# Counting triplet IDs to calculate number of occurences
		for x in "${parts[@]}"
		do
			triplet_part_file=$paths_folder"/triplet_id_"$x
			triplet_count_file=$paths_folder"/triplet_count_"$x
			( awk -F "\t" '{relations[$0]++} END{for(relation in relations){print relation"\t"relations[relation]}}' $triplet_part_file > $triplet_count_file ) &
		done
		wait

		rm $paths_folder"/triplet_id_"*

		cat $paths_folder"/triplet_count_"* > $paths_folder"/triplet_count";

		rm $paths_folder"/triplet_count_"*

		# Creating a triplet occurence matrix
		gawk -F $'\t' '{ matrix[$1][$2][$3]+=$4; } END{for (x in matrix) {for (y in matrix[x]) {for (path in matrix[x][y]) {print x, y, path, matrix[x][y][path]}}}}' $paths_folder"/triplet_count" > $paths_folder"/final_count"

		rm $paths_folder"/triplet_count"

		python3 path_terms_indexer.py $paths_folder $paths_folder"/final_count" $prefix 3;

		rm $paths_folder"/final_count"
	done

done

rm $corpus"_split_"*