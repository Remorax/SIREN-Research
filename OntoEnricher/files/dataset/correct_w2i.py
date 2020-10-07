import sys, time, pickledb
import spacy, subprocess, itertools, multiprocessing, sys, glob,  en_core_web_lg, neuralcoref
from spacy.tokens.token import Token
from spacy.attrs import ORTH, LEMMA
from collections import Counter


def preprocess_word(noun):
    filt_tokens = ["DET", "ADV", "PUNCT", "CCONJ"]
    start_index = [i for i,token in enumerate(noun) if token.pos_ not in filt_tokens][0]
    np_filt = noun[start_index:].text
    if "(" not in np_filt and ")" in np_filt:
        np_filt = np_filt.replace(")", "")
    elif "(" in np_filt and ")" not in np_filt:
        np_filt = np_filt.replace("(", "")
    return np_filt


nlp = en_core_web_lg.load()

prefix = "../junk/db_files/"

word2id_db_corrected = pickledb.load(prefix + "w2i_corrected.db", True)
id2word_db_corrected = pickledb.load(prefix + "i2w_corrected.db", True)

def p(word):
    try:
        return preprocess_word(word)
    except KeyboardInterrupt:
        sys.exit()
        pass
    except Exception:
        return word.text

idx = 0

window_size = 1000000
batches = int(len(allkeys)/window_size)
for i in range(batches):
    piped_words = list(nlp.pipe(allkeys[i*window_size: (i+1)*window_size]))
    for corrected_key in piped_words:
        word = p(corrected_key)
        word2id_db_corrected[word] = str(idx)
        id2word_db_corrected[str(idx)] = word
        idx += 1

# print (time.time()-t)