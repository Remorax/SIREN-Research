dataset="../files/dataset/dataset.tsv"
split -l 50 $dataset $dataset"_split_" --numeric-suffixes=1;
array=(../files/dataset/dataset.tsv_split_*)
i=0
for current in "${array[@]}";
do
        i=$((i+1))
	echo $current
        if [ $(($i%100)) -eq 0 ]
        then
                echo $i
                wait
        else
                ( python3 dataset_creator.py $current ) &
        fi
done
wait
rm $dataset"_split_"*