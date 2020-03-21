import spacy, subprocess, itertools, multiprocessing
from spacy.tokens.token import Token

MAX_PATH_LEN = 6


def stringifyEdge(word, root=True):
    try:
        w = word.root
    except:
        w = word

    if isinstance(word, Token):
        word = word.lemma_.strip().lower()
    else:
        word = ' '.join([wd.string.strip().lower() for wd in word])
    pos, deps = w.pos_, w.dep_
    path = '/'.join([word, pos, deps if deps and root else 'ROOT'])
    return path

def stringifyArg(word, edge):
    try:
        word = word.root
    except:
        pass
    pos, deps = word.pos_, word.dep_
    path = '/'.join([edge, pos, deps if deps else 'ROOT'])
    return path

def filterPaths(function, lowestCommonHead, paths):
    path1 = [lowestCommonHead]
    path1.extend(paths[:-1])
    path2 = paths
    return any(node not in function(path) for path, node in list(zip(path1, path2)))

def notPunct(arr):
    firstWord = arr[0]
    return firstWord.tag_ != 'PUNCT' and len(firstWord.string.strip()) > 1

def notEqual(x, y):
    try:
        return x!=y
    except:
        return False

def checkHead(token, lowestCommonHead):
    return isinstance(token, Token) and lowestCommonHead == token

def getPathFromRoot(phrase):
    paths = []
    head = phrase.head
    while phrase != head:
        phrase = phrase.head
        paths.append(phrase)
        head = phrase.head
    paths = paths[::-1]
    return paths

def breakCompoundWords(elem):
    try:
        root = elem.root
        return root
    except:
        return elem

def findMinLength(x, y):
    if len(x) < len(y):
        return (len(x), x)
    return (len(y), y)

def findLowestCommonHead(pathX, pathY, minLength, minArray):
    lowestCommonHead = None
    if minLength:        
        uncommon = [i for i in range(minLength) if pathX[i] != pathY[i]]
        if uncommon:
            idx = uncommon[0] - 1
        else:
            idx = minLength - 1
        lowestCommonHead = minArray[idx]
    else:
        idx = 0
        if pathX:
            lowestCommonHead = pathX[0]
        elif pathY:
            lowestCommonHead = pathY[0]
        else:
            lowestCommonHead = None
    
    return idx, lowestCommonHead

def getShortestPath(tup):

    xinit, yinit = tup[0], tup[1]

    x, y = breakCompoundWords(xinit), breakCompoundWords(yinit)
    
    pathX, pathY = getPathFromRoot(x), getPathFromRoot(y)
    
    minLength, minArray = findMinLength(pathX, pathY)
    
    idx, lowestCommonHead = findLowestCommonHead(pathX, pathY, minLength, minArray)
    
    try:
        pathX = pathX[idx+1:]
        pathY = pathY[idx+1:]
        checkLeft, checkRight = lambda h: h.lefts, lambda h: h.rights
        if lowestCommonHead and (filterPaths(checkLeft, lowestCommonHead, pathX) or filterPaths(checkRight, lowestCommonHead, pathY)):
            return None
        pathX = pathX[::-1]

        paths = [(None, xinit, pathX, lowestCommonHead, pathY, yinit, None)]
        lefts, rights = list(xinit.lefts), list(yinit.rights)

        if lefts and notPunct(lefts):
            paths.append((lefts[0], xinit, pathX, lowestCommonHead, pathY, yinit, None))

        if rights and notPunct(rights):
            paths.append((None, xinit, pathX, lowestCommonHead, pathY, yinit, rights[0]))
        
        return paths
    except Exception as e:
        print (e)
        return None

def stringifyFilterPath(path):

    lowestCommonHeads = []
    (leftX, x, pathX, lowestCommonHead, pathY, y, rightY) = path

    isXHead, isYHead = checkHead(x, lowestCommonHead), checkHead(y, lowestCommonHead)
    signX = '' if isXHead else '>'
    leftXPath  = []
    if leftX:
        edge_str = stringifyEdge(leftX)
        leftXPath.append(edge_str + "<")

    signY = '' if isYHead else '<'
    rightYPath = []
    if rightY:
        edge_str = stringifyEdge(rightY)
        rightYPath.append(">" + edge_str)

    lowestCommonHeads = [[stringifyEdge(lowestCommonHead, False)] if lowestCommonHead and not (isYHead or isXHead) else []][0]
    
    if MAX_PATH_LEN >= len(pathX + leftXPath + pathY + rightYPath + lowestCommonHeads):
        
        if isinstance(x, Token):
            stringifiedX = x.string.strip().lower()
        else:
            stringifiedX = ' '.join([x_wd.string.strip().lower() for x_wd in x])
        
        if isinstance(y, Token):
            stringifiedY = y.string.strip().lower()
        else:
            stringifiedY = ' '.join([y_wd.string.strip().lower() for y_wd in y])

        stringifiedPathX, stringifiedPathY = [stringifyEdge(word) + ">" for word in pathX], ["<" + stringifyEdge(word) for word in pathY]
        stringifiedArgX, stringifiedArgY = [stringifyArg(x, 'X') + signX], [signY + stringifyArg(y, 'Y')]
        
        stringifiedPath = '_'.join(leftXPath + stringifiedArgX + stringifiedPathX + lowestCommonHeads + stringifiedPathY + stringifiedArgY + rightYPath)

        return (stringifiedX, stringifiedY, stringifiedPath)

    return None

def getDependencyPaths(sentence, nlp, sentenceNounChunks):

    nps = [(n, n.start, n.end) for n in sentenceNounChunks]
    nps.extend([(word, pos, pos) for (pos, word) in enumerate(sentence) if word.tag_[:2] == 'NN' and len(word.string.strip()) > 2])
    ls = list(itertools.product(nps, nps))
    pairedConcepts = [(el[0][0], el[1][0]) for el in itertools.product(nps, nps) if el[1][1] > el[0][2] and notEqual(el[0], el[1])]
    pairedConcepts = list(dict.fromkeys(pairedConcepts))
    
    paths = []
    for pair in pairedConcepts:
        appendingElem = getShortestPath(pair)
        if appendingElem:
            paths.extend([stringifyFilterPath(path) for path in appendingElem])

    return paths

def splitFile (file, n):
    inputfile = open(file, 'r')
    output = None
    suffix = 0
    for (i, line) in enumerate(inputfile):
        if i % n == 0:
            if output:
                output.close()
            output = open(file + "_split_" + str(suffix) + '.txt', 'w+')
            suffix += 1
        output.write(line)
    output.close()
    return suffix

def parseText(idx):
    global file

    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe(nlp.create_pipe('sentencizer'), before="parser")
    fileName = file + "_split_" + str(idx) + ".txt"
    op = file + "_parsed_" + str(idx)

    with open(fileName, "r") as inp:
        with open(op, "w+") as out:
            for i,para in enumerate(inp):
                if not para.strip(): continue
                nounChunks = list(nlp(para).nounChunks).copy()
                sentences = nlp(para.strip()).sents
                for sentence in sentences:
                    if "<doc id=" in sentence.text or "</doc>" in sentence.text:
                        continue
                    sentenceNounChunks = [n for n in nounChunks if sentence.start <= n.start < n.end - 1 < sentence.end]
                    dependencies = getDependencyPaths(sentence, nlp, sentenceNounChunks)
                    if dependencies:
                        allpaths = ["\t".join(path) for path in dependencies if path]
                        out.write("\n".join(allpaths))


if __name__ == "__main__":    
    file = "../junk/temp"
    countlines = "wc -l " + file 
    output, _ = subprocess.Popen(countlines.split(), stdout=subprocess.PIPE).communicate()
    n = int(output.decode("utf-8").strip().split(" ")[0]) + 1
    m = int(n/20)
    suffix = splitFile(file, m)
    processes = []
    for i in range(20):
        p = multiprocessing.Process(target=parseText, args=(i,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    
