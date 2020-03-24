#!/usr/bin/env bash

corpus=$1 
folder=$2
prefix=$3

n=`grep "" -c $corpus | awk '{ print $1 }'`
n=$((($n + 1)/5))


gsplit -l $n $corpus $corpus"_split_" --numeric-suffixes=1;

for x in {01..05}
do
	corpus_part=$corpus"_split_"$x
	( python3 corpus_parser.py $corpus_part ) &
done
wait


parsed_final=$corpus"_parsed"
cat $corpus"_split_"*"_parsed" > $parsed_final


for x in {01..05}
do
	parsed_final_part=$corpus"_split_"$x"_parsed"
	( awk -F "\t" '{relations[$3]++} END{for(relation in relations){print relation"\t"relations[relation]}}' $parsed_final_part > $corpus"_paths_"$x ) &
done
wait
echo "Done till here.. part 3"

paths=$folder"all_paths"
cat $corpus"_paths_"* > $paths
rm $corpus"_paths_"*

# declare -a path_thresholds=(3 7 10 15 20 25 50)
declare -a path_thresholds=(3 5)
for n in "${path_thresholds[@]}"
do
	mkdir $folder$prefix"_threshold_"$n
	( awk -F "\t" '{i[$1]+=$2} END{for(x in i){ if (i[x] >= '$n') print x } }' $paths > $folder$prefix"_threshold_"$n'/filtered_paths'  ) &
done
wait
rm $paths

# echo "Done till here.. part 4"

# # Create the terms file
awk -F$'\t' '{if (a[$1] == 0) {a[$1] = -1; print $1}}' $parsed_final > $output"xterms" & PIDLEFT=$!
awk -F$'\t' '{if (a[$2] == 0) {a[$2] = -1; print $2}}' $parsed_final > $output"yterms" & PIDRIGHT=$!

# echo "Done till here.. part 5"

wait $PIDLEFT
wait $PIDRIGHT
cat $output"xterms" $output"yterms" | sort -u > $output"terms";
rm $output"xterms" $output"yterms" $parsed_final

# # First step - create the term and path to ID dictionaries
echo 'Creating the resource from the triplets file...'
for n in "${path_thresholds[@]}"
do
	paths_folder=$folder$prefix"_threshold_"$n
	( python3 path_terms_indexer.py $paths_folder $output"terms" $prefix; ) &
done
wait

# # Second step - convert the textual triplets to triplets of IDs. 
# for x in {a..t}
# do
# ( python create_resource_from_corpus_2.py $wiki_dump_file"_a"$x"_parsed" $resource_prefix ) &
# done
# wait

# # Third step - use the ID-based triplet file and converts it to the '_l2r.db' file
# for x in {a..t}
# do
# ( awk -v OFS='\t' '{i[$0]++} END{for(x in i){print x, i[x]}}' $wiki_dump_file"_a"$x"_parsed_id" > id_triplet_file"_a"$x ) &
# done
# wait

# cat id_triplet_file_* > "id_triplet_file_temp";

# # ls 

# if test -f "id_triplet_file_temp"; then
# 	echo "id_triplet_file_temp exists"
# fi

# for x in {0..4}
# do
# ( gawk -F $'\t' '{ if($1%5==$x) {a[$1][$2][$3]+=$4; } } END {for (i in a) for (j in a[i]) for (k in a[i][j]) print i, j, k, a[i][j][k]}' id_triplet_file_temp > id_triplet_file_$x ) &
# done
# wait
# # ls
# if test -f "id_triplet_file_temp"; then
# 	echo "id_triplet_file_temp exists"
# fi


# cat id_triplet_file_* > id_triplet_file;
# # ls
# rm id_triplet_file_temp id_triplet_file_* $triplet_file"_"*;

# python create_resource_from_corpus_3.py id_triplet_file $resource_prefix;

# # You can delete triplet_file now and keep only id_triplet_file which is more efficient, or delete both.

rm $corpus"_split_"*