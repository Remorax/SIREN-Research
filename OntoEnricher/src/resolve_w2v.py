wiki2vec = KeyedVectors.load_word2vec_format("/home/vlead/enwiki_20180420_win10_300d.txt")
og_dict = deepcopy(wiki2vec.wv.vocab)
for k in og_dict:
    if "/" in k:
        wiki2vec.wv.vocab[k.split("/")[1].lower()] = wiki2vec.wv.vocab[k]
        del wiki2vec.wv.vocab[k]

del og_dict
def id_to_entity(db, entity_id):
    entity = db.get(str(entity_id))
    if not entity:
        print (entity_id)
    return entity
def closest_word_w2v(word1):
    len_part = 100000
    max_sim = -1000
    n_parts = ceil(len(words_filt)/len_part)
    closest_word = ""
    if word1 not in wiki2vec.wv.vocab:
        return (word1, closest_word, max_sim)
    for i in range(n_parts):
        words_part = words_filt[i*len_part:(i+1)*len_part]
        closest_word, max_sim = calculate_sim(words_part, word1, max_sim, closest_word)
    with counter.get_lock():
        counter.value += 1

    print ("Original word: ", word1, "Closest Word: ", closest_word, "Sim: ", max_sim)
    print ("Percentage done: ", float(counter.value*100/len(failed)))
    return (word1, closest_word, max_sim) 

def id_to_path(db, entity_id):
    entity = db.get(str(entity_id))
    entity = "/".join(["*##*".join(e.split("_", 1)) for e in entity.split("/")])
    return entity
def calculate_sim(words, word1, max_sim, closest_word):
        i = 0
        for word2 in words:
            try:
                sim = wiki2vec.similarity("_".join(word1.lower().split()), "_".join(word2.split()))
                if sim > max_sim:
                    max_sim = sim
                    closest_word = word2
                i += 1
            except Exception as e:
                continue
        return (closest_word, max_sim)

def construct_resolved():
    global failed, words_db
    words_filt = []


    print ("filtering words...")
    for w in words_db:
        try:
            s = wiki2vec[w]
            words_filt.append(w)
        except:
            continue
    print ("Filtered from {} to {}".format(len(words_db), len(words_filt)))

    a = time.time()
    counter = Value('i', 0)
    print ("Working on it...")
    resolved = dict()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for res in executor.map(closest_word_w2v, failed):
            print (res)
            resolved[res[0]] = (res[1], res[2])
    return resolved

def entity_to_id(db, entity, resolved):
    global success, failed
    entity_id = db.get(entity)
    if entity_id:
        success.append(entity)
        return int(entity_id)
    closest_entity = resolved.get(entity, "")
    if closest_entity and closest_entity[0]:
        success.append(entity)
        return int(db.get(closest_entity[0]))
    failed.append(entity)
    return -1

def entity_to_id_unresolved(db, entity):
    global success, failed
    entity_id = db.get(entity)
    if entity_id:
        success.append(entity)
        return int(entity_id)
    failed.append(entity)
    return -1

def extract_paths(db, x, y):
    key = (str(x) + '###' + str(y))
    try:
        relation = db.get(key)
        return {int(path_count.split(":")[0]): int(path_count.split(":")[1]) for path_count in relation.split(",")}
    except Exception as e:
        return {}

def load_embeddings_from_disk():
    try:
        vectors = bcolz.open(embeddings_folder)[:]
        words = pickle.load(open(embeddings_folder + 'words.pkl', 'rb'))
        word2idx = pickle.load(open(embeddings_folder + 'words_index.pkl', 'rb'))

        embeddings = vectors
    except:
        embeddings, word2idx = create_embeddings()
    return embeddings, word2idx


def create_embeddings():
#     print (success)
    vocab = set([a for a in success if a])
    words = ['_unk_']
    idx = 1
    word2idx = {"_unk_": 0}
    vectors = bcolz.carray(np.random.uniform(-1, 1, (1, 300)), rootdir=embeddings_folder, mode='w')
    with open(embeddings_file, 'r') as f:
        for l in f:
            line = [a[::-1] for a in l[::-1].split(" ", 300)[::-1]]
            word, vector = line[0], [float(s) for s in line[1:]]
            if len(vector) != 300:
                print (len(vector))
            if word not in vocab:
                continue
            words.append(word)
            vectors.append(np.resize(np.array(vector), (1, 300)).astype(np.float))
            word2idx[word] = idx
            idx += 1
#     print (vectors.shape)
    row_norm = np.sum(np.abs(vectors)**2, axis=-1)**(1./2)
    vectors /= row_norm[:, np.newaxis]
    vectors = bcolz.carray(vectors, rootdir=embeddings_folder, mode='w')
    vectors.flush()

    pickle.dump(words, open(embeddings_folder + 'words.pkl', 'wb'))
    pickle.dump(word2idx, open(embeddings_folder + 'words_index.pkl', 'wb'))

    return vectors, word2idx
def extract_direction(edge):

    if edge[0] == ">" or edge[0] == "<":
        direction = "start_" + arrow_heads[edge[0]]
        edge = edge[1:]
    elif edge[-1] == ">" or edge[-1] == "<":
        direction = "end_" + arrow_heads[edge[-1]]
        edge = edge[:-1]
    else:
        direction = ' '
    return direction, edge

def parse_path(path):
    parsed_path = []
    for edge in path.split("*##*"):
        direction, edge = extract_direction(edge)
        if edge.split("/"):
            try:
                embedding, pos, dependency = tuple([a[::-1] for a in edge[::-1].split("/",2)][::-1])
            except:
                print (edge, path)
                raise
            emb_idx, pos_idx, dep_idx, dir_idx = emb_indexer.get(embedding, 0), pos_indexer[pos], dep_indexer[dependency], dir_indexer[direction]
            parsed_path.append(tuple([emb_idx, pos_idx, dep_idx, dir_idx]))
        else:
            return None
    return tuple(parsed_path)

def parse_tuple(tup):
    x, y = tup
#     paths = list(extract_paths(relations_db,x,y).items()) + list(extract_paths(relations_db,y,x).items())
#     x_word = id_to_entity(id2word_db, x) if x!=-1 else "X"
#     y_word = id_to_entity(id2word_db, y) if y!=-1 else "Y"
#     path_count_dict = { id_to_path(id2path_db, path).replace("X/", x_word+"/").replace("Y/", y_word+"/") : freq for (path, freq) in paths }
    paths_xy = list(extract_paths(relations_db,x,y).items())
    paths_yx = list(extract_paths(relations_db,y,x).items())
    path_count_dict = { id_to_path(id2path_db, path) : freq for (path, freq) in paths_xy }
    path_count_dict.update({ id_to_path(id2path_db, path).replace("X/", '@@@').replace('Y/', 'X/').replace('@@@', 'Y/') : freq for (path, freq) in paths_yx })
    return path_count_dict

def parse_dataset(dataset):
    print ("Parsing dataset for ", prefix)
    _ = [(entity_to_id_unresolved(word2id_db, tup[0]), entity_to_id_unresolved(word2id_db, tup[1])) for tup in dataset]
    resolved = construct_resolved()
    global embeddings, emb_indexer
    success, failed = [], []
    dataset = [(entity_to_id(word2id_db, tup[0], resolved), entity_to_id(word2id_db, tup[1], resolved)) for tup in dataset]
    embeddings, emb_indexer = create_embeddings()
    
    parsed_dicts = [parse_tuple(tup) for tup in dataset]
    parsed_dicts = [{ parse_path(path) : path_count_dict[path] for path in path_count_dict } for path_count_dict in parsed_dicts]
    paths = [{ path : path_count_dict[path] for path in path_count_dict if path} for path_count_dict in parsed_dicts]
    empty = [list(dataset)[i] for i, path_list in enumerate(paths) if len(list(path_list.keys())) == 0]
#     paths = [{NULL_PATH: 1} if not path_list else path_list for i, path_list in enumerate(paths)]
    embed_indices = [(emb_indexer.get(x,0), emb_indexer.get(y,0)) for (x,y) in dataset]

    return embed_indices, paths


embeddings_folder = "../junk/Wiki2vec_lite.dat/"
POS_DIM = 4
DEP_DIM = 5
DIR_DIM = 1
EMBEDDING_DIM = 300
relations = ["hypernym", "hyponym", "concept", "instance", "none"]
NUM_RELATIONS = len(relations)
NULL_PATH = ((0, 0, 0, 0),)

for dbpedia_neg in range(10, 101, 10):

    true_file = open("../files/dataset/dataset_t.tsv", "r").read().split("\n")
    false_file = open("../files/dataset/dataset_f.tsv", "r").read().split("\n")
    false_file = false_file[:int(len(false_file) * float(dbpedia_neg/100))]

    custom_train = true_file[:int(0.9*len(true_file))] + false_file[:int(0.9*len(false_file))]
    custom_test = true_file[int(0.9*len(true_file)):] + false_file[int(0.9*len(false_file)):]
        
    success, failed = [], []

    word2id_db = pickledb.load(prefix + "w2i.db", False)
    id2word_db = pickledb.load(prefix + "i2w.db", False)
    path2id_db = pickledb.load(prefix + "p2i.db", False)
    id2path_db = pickledb.load(prefix + "i2p.db", False)
    relations_db = pickledb.load(prefix + "relations.db", False)

    words_db = word2id_db.getall()
    
    op_file = "../junk/dataset_parsed_wiki2vec_new_" + str(dbpedia_neg) + ".pkl"

    train_dataset = {tuple(l.split("\t")[:2]): l.split("\t")[2] for l in custom_train}
    test_dataset = {tuple(l.split("\t")[:2]): l.split("\t")[2] for l in custom_test}
    test_instances = {tuple(l.split("\t")[:2]): l.split("\t")[2] for l in open(instances_file).read().split("\n")}
    test_knocked = {tuple(l.split("\t")[:2]): l.split("\t")[2] for l in open(knocked_file).read().split("\n")}

    arrow_heads = {">": "up", "<":"down"}

    pos_indexer, dep_indexer, dir_indexer = defaultdict(count(0).__next__), defaultdict(count(0).__next__), defaultdict(count(0).__next__)
    unk_pos, unk_dep, unk_dir = pos_indexer["#UNKNOWN#"], dep_indexer["#UNKNOWN#"], dir_indexer["#UNKNOWN#"]

    dataset_keys = list(train_dataset.keys()) + list(test_dataset.keys()) + list(test_instances.keys()) + list(test_knocked.keys())
    dataset_vals = list(train_dataset.values()) + list(test_dataset.values()) + list(test_instances.values()) + list(test_knocked.values())

    embeddings, emb_indexer = None, None

    mappingDict = {key: idx for (idx,key) in enumerate(relations)}

    embed_indices, x = parse_dataset(dataset_keys)
    y = [mappingDict[relation] for relation in dataset_vals]


    s1 = len(train_dataset)
    s2 = len(train_dataset) + len(test_dataset)
    s3 = len(train_dataset)+len(test_dataset)+len(test_instances)

    parsed_train = (embed_indices[:s1], x[:s1], y[:s1], dataset_keys[:s1], dataset_vals[:s1])
    parsed_test = (embed_indices[s1:s2], x[s1:s2], y[s1:s2], dataset_keys[s1:s2], dataset_vals[s1:s2])
    parsed_instances = (embed_indices[s2:s3], x[s2:s3], y[s2:s3], dataset_keys[s2:s3], dataset_vals[s2:s3])
    parsed_knocked = (embed_indices[s3:], x[s3:], y[s3:], dataset_keys[s3:], dataset_vals[s3:])

    f = open(op_file, "wb+")
    pickle.dump([parsed_train, parsed_test, parsed_instances, parsed_knocked, pos_indexer, dep_indexer, dir_indexer], f)
    f.close()

    print ("Successful hits: ", len(success), "Failed hits: ", len(failed))
    print ("Parsed",prefix) 