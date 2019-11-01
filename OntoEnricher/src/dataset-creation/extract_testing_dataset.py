import re
import urllib.request
import spacy, neuralcoref
from bs4 import BeautifulSoup
from subject_verb_object_extract import findSVOs, nlp

# thresholdWord = "pizza"
# url = "https://www.webstaurantstore.com/article/101/types-of-pizza.html"
# className = "section"

thresholdWord = "cybersecurity"
url = "https://tools.cisco.com/security/center/resources/virus_differences"
className = "sitecopy"

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
            # print (i.leaves())
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
    return current_chunk

def getSim(a, b):
    try:
        return model.similarity(a, b)
    except:
        return 0

def filter_dissimilar(ls):
    newls = []
    for elem in ls:
        if elem[1] >= 0.2:
            newls.append(elem[0])
        elif elem[1] == 0:
            sim = sum([getSim(el, thresholdWord) for el in elem[0].split()])
            if sim > 0.2:
                newls.append(elem[0])
    return newls

def multiwordFilter(a,b):
#     aword = "_".join(a.split(" "))
#     if aword not in model.wv.vocab:
#         aword = a.split(" ")[-1]
    
#     bword = "_".join(a.split(" "))
#     if bword not in model.wv.vocab:
#         bword = b.split(" ")[-1]
    
    if getSim(a,b) >= 0.2:
        return True
    return False
 
html = urllib.request.urlopen()
soup = BeautifulSoup(html)
data = soup.find("div", {"class": className})
paras = data.findAll("p")
paras = [o.text for o in paras]

nlp = spacy.load('en_core_web_lg')

# load NeuralCoref and add it to the pipe of SpaCy's model
coref = neuralcoref.NeuralCoref(nlp.vocab)
nlp.add_pipe(coref, name='neuralcoref')

paras = [nlp(para)._.coref_resolved for para in paras]


# For SVO extraction: less accurate
# allsvos = []
# for para in paras:
#     tokens = nlp(sent)
#     svos = findSVOs(tokens)
#     allsvos.extend(svos)


testData = []
for para in paras:
    instances = getInstances(para)
    ls = [(word, getSim("_".join(word.split(" ")), thresholdWord)) for word in list(set([a.lower() for a in instances]))]
    ls = list(set(list(itertools.combinations(filter_dissimilar(ls), 2))))
    testData.extend([[a,b] for (a,b) in ls if multiwordFilter(a,b)])
    
testData_uniq = [el.split("\t") for el in list(set(["\t".join(el) for el in testData]))]

testData2 = [el + [getSim(el[0],el[1])] for el in testData_uniq]
testData2.sort(key=lambda x:x[2], reverse=True)

open("test_" + thresholdWord + "_1000.tsv","w+").write("\n".join(["\t".join(el[:2]) for el in testData2[:100]]))
