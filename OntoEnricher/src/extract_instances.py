import re, glob
import urllib.request,  en_core_web_sm
import spacy, neuralcoref, itertools
from bs4 import BeautifulSoup
from bs4.element import Comment
from subject_verb_object_extract import findSVOs, nlp
from nltk.chunk.regexp import RegexpParser
from nltk import pos_tag, word_tokenize, sys
from nltk.tree import Tree

def getInstances(text):
    grammar = """
        PRE:   {<NNS|NNP|NN|NP|JJ|UH>+}
        INSTANCE:   {(<JJ+>)?<PRE>}
    """
    chunker = RegexpParser(grammar)
    taggedText = pos_tag(word_tokenize(text))
    textChunks = chunker.parse(taggedText)
    current_chunk = []
    for i in textChunks:
        if (type(i) == Tree and i.label() == "INSTANCE"):
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
    return current_chunk


nlp = en_core_web_sm.load()


# load NeuralCoref and add it to the pipe of SpaCy's model, for coreference resolution
coref = neuralcoref.NeuralCoref(nlp.vocab)
nlp.add_pipe(coref, name='neuralcoref')

for i,file in enumerate(glob.glob("../files/dataset/security*")):
    paras = [l for l in open(file).read().split("\n") if l]
    print (file)
    sys.stdout.flush()
    paras = [nlp(para)._.coref_resolved for para in paras]
    print ("done")
    testData = []
    for para in paras:
        instances = getInstances(para)
        ls = list(set(instances))
        ls = list(set(list(itertools.combinations(ls, 2))))
        testData.extend(["\t".join([a,b]) for (a,b) in ls])
    
    testData = [el + "\tnone" for el in list(set(testData))]

    open("../files/dataset/instances" + str(i) + ".tsv", "w+").write("\n".join(testData))

