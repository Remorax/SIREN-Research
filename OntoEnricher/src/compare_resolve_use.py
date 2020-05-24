import os, pickle
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import threading
import subprocess
import re

t = time.time()
def write(statement):
    op_file = open("Logs", "a+")
    op_file.write("\n" + str(statement) + "\n")
    op_file.close()

def stats():
    t = threading.Timer(60, stats)
    t.daemon=True
    t.start()
    bashCommand1 = "ps -ef"
    process1 = subprocess.Popen(bashCommand1.split(), stdout=subprocess.PIPE)
    output, error = process1.communicate()
    output = re.findall(".* python3.*", output.decode("utf-8"))
    write("\n".join(output))

stats()


USE_folder = "/home/vlead/USE"

f = open("../junk/failed_instances", "rb")
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
    # print ("Original word: ", wd, "Closest Word: ", closest_word, "Sim: ", max_sim)
    return (wd, closest_word, max_sim)

def extractUSEEmbeddings(words):
    embed = hub.KerasLayer(USE_folder)
    word_embeddings = embed(words)
    return word_embeddings.numpy()

def run():
    failed_embeds = extractUSEEmbeddings(failed)

    print ("Extracted failed embeddings")

    for file in ["../junk/" + s for s in os.listdir("../junk/") if s.startswith("use_embeddings_")]:
        emb_file = open(file, "rb")
        use_embeds = pickle.load(emb_file)
        output = {}
        args = [(word, failed_embeds[i], use_embeds) for i,word in enumerate(failed)]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for result in executor.map(compare, args):
                output[result[0]] = (result[1], result[2])
        results = {word: results[i] if results[i][1] > output[i][1] else output[i] for i in results}
        print ("Parsing done for", file)

    resolved_file = open("../junk/resolved_use.pkl", "wb")
    pickle.dump(results, resolved_file)

    write("Time taken for execution: {}".format(time.time() - t))

if __name__ == '__main__':
    run()
