import sys
from bsddb3 import btopen

def indexPathTerm(words_file):
    path_to_id = btopen(paths_folder + "/" + prefix + '_path_to_id.db', 'c')
    id_to_path = btopen(paths_folder + "/" + prefix + '_id_to_path.db', 'c')
    word_to_id = btopen(paths_folder + "/" + prefix + '_word_to_id.db', 'c')
    id_to_word = btopen(paths_folder + "/" + prefix + '_id_to_word.db', 'c')    

    with open(paths_folder + "/filtered_paths", encoding="utf-8") as paths:
        
        filtered_paths = []
        for path in paths:
            filtered_paths.append(path.strip())
        filtered_paths = list(set(filtered_paths))


        path_to_id_dict = {filtered_paths[i]:i for i in range(len(filtered_paths))}


        for path, path_id in path_to_id_dict.items():
            path_id, path = str(path_id).encode("utf-8"), str(path).encode("utf-8")
            path_to_id[path], id_to_path[path_id] = path_id, path

        path_to_id.sync()
        id_to_path.sync()

    with open(words_file, "r", encoding="utf-8") as terms:
        words = []
        for term in terms:
            words.append(term.strip())

        word_to_id_dict = {words[i]:i for i in range(len(words))}

        for word, word_id in word_to_id_dict.items():
            word_id, word = str(word_id).encode("utf-8"), str(word).encode("utf-8")
            word_to_id[word], id_to_word[word_id] = word_id, word

        word_to_id.sync()
        id_to_word.sync()


def getTripletIDFromDB(parsed_file):
    word_to_id = btopen(paths_folder + "/" + prefix + '_word_to_id.db')
    path_to_id = btopen(paths_folder + "/" + prefix + '_path_to_id.db')

    x = file.split("_")[-2]
    output_parsed = open(paths_folder + '/triplet_id_' + x, 'w+')

    with open(parsed_file) as parsed_inp:
        for line in parsed_inp:
            if line.strip():
                x, y, path = line.strip().split('\t')

            x, y = x.strip().encode("utf-8"), y.strip().encode("utf-8")
            x_id, y_id = str(word_to_id[x].decode("utf-8")), str(word_to_id[y].decode("utf-8"))
            
            path_id = path_to_id.get(path.strip().encode("utf-8"), -1)
            if path_id != -1:
                path_id = str(path_id.decode("utf-8"))
                triplet = "\t".join((x_id, y_id, path_id))
                output_parsed.write(triplet + "\n")

    output_parsed.close()

def indexWordPairs(parsed_file):

    word_occurence_map = btopen(paths_folder + "/" + prefix + '_word_occurence_map.db', 'c')

    with open(file) as inp:
        for line in inp:
            x, y, path, count = line.strip().split('\t')

            key = str(x) + '_' + str(y)
            key = key.encode("utf-8")
            current = path + ":" + count

            if key in word_occurence_map:
                pastkeys = word_occurence_map[key].decode('utf-8')
                current =  pastkeys + current
            
            current = current.encode("utf-8")
            
            word_occurence_map[key] = current

    word_occurence_map.sync()



paths_folder = sys.argv[1]
file = sys.argv[2]
prefix = sys.argv[3]
mode = sys.argv[4]
if mode=="1":
    indexPathTerm(file)
elif mode=="2":
    getTripletIDFromDB(file)
elif mode=="3":
    indexWordPairs(file)

    
