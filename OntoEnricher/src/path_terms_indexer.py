import bsddb3, sys


paths_folder = sys.argv[1]
words_file = sys.argv[2]
prefix = sys.argv[3]

path_to_id = bsddb3.btopen(paths_folder + "/" + prefix + '_path_to_id.db', 'c')
id_to_path = bsddb3.btopen(paths_folder + "/" + prefix + '_id_to_path.db', 'c')
word_to_id = bsddb3.btopen(paths_folder + "/" + prefix + '_term_to_id.db', 'c')
id_to_word = bsddb3.btopen(paths_folder + "/" + prefix + '_id_to_term.db', 'c')    

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

