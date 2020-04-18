array=(security_threshold*/*_word_occurence_map.db)
i=0
for current in "${array[@]}";
do
        i=$((i+1))
        if [ $(($i%80)) -eq 0 ]
        then
                echo $i
                wait
        else
                ( python3 convert_to_shelve.py $current ) &
        fi
done
wait
