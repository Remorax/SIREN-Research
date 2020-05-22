import os, pickle
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

USE_folder = "/home/vlead/USE"

f = open("../junk/failed_words", "rb")
failed, _ = pickle.load(f)
results = {i: ("", -1000) for i in failed}

def compare(wd, word, embeds):
    max_sim = -1000
    closest_word = ""
    for w in embeds:
        sim = np.dot(word, w[1])
        if sim > max_sim:
            max_sim = sim
            closest_word = w[0]
    print ("Original word: ", wd, "Closest Word: ", closest_word, "Sim: ", max_sim)
    return (closest_word, max_sim)

def extractUSEEmbeddings(words):
    embed = hub.KerasLayer(USE_folder)
    word_embeddings = embed(words)
    return word_embeddings.numpy()

failed_embeds = extractUSEEmbeddings(failed)

print ("Extracted failed embeddings")

for file in ["../junk/" + s for s in os.listdir("../junk/") if s.startswith("use_embeddings_")]:
    emb_file = open(file, "rb")
    use_embeds = pickle.load(emb_file)
    output = {word: compare(word, failed_embeds[i], use_embeds) for i,word in enumerate(failed)}
    results = {word: results[i] if results[i][1] > output[i][1] else output[i] for i in results}
    print ("Parsing done for", file)

resolved_file = open("../junk/resolved_use.pkl", "wb")
pickle.dump(results, resolved_file)
