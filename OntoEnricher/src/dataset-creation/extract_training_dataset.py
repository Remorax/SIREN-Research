from pronto import Ontology
from re import finditer
from SPARQLWrapper import SPARQLWrapper, JSON
from pywikibot.data import api
import pywikibot, urllib.request, json, os
from nltk.corpus import wordnet as wn
from gensim.models import KeyedVectors

domainName = "security" # Just a naming convention

model = KeyedVectors.load_word2vec_format("~/GoogleNews-vectors-negative300.bin", binary=True)

# APPROACH 1: CONCEPT EXTRACTION


# Creation of wordsList of "concepts" by parsing ontologies
# Ignore if already parsed

def camel_case_split(identifier):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return " ".join([m.group(0) for m in matches])

def extractHypernymsFromOntology(ontology):    
    ont = Ontology(ontology)
    allConcepts = []
    listid = []
    dictelem = {}
    for term in ont:
        allConcepts.append(term)
        if term.children:
            a = str(term).split(":")
            b = a[0]
            listid.append(b[1:])
    for x in range(0,len(listid)):
        key = listid[x]
        if key in dictelem:
            child = ont[listid[x].children].split(":")
            ch = child[0]
            dictelem.get(key).append(ch[1:])
        else:
            childs = ont[listid[x]].children
            all_childs = ""
            for y in childs:
                z = str(y).split(":")
                f = z[0]
                all_childs += f[1:]+","
            dictelem[key] = all_childs

    
    finalDict = {}

    for elem in dictelem:
        newelem = camel_case_split(elem)
        ls = dictelem[elem].split(",")[:-1]
        newval = ",".join([camel_case_split(el) for el in ls])
        finalDict[newelem] = newval

    hypernymsList = []
    for elem in finalDict:
        hypernymsList.extend([(elem, val) for val in finalDict[elem].split(",")])
    
    return (hypernymsList, allConcepts)

def parseOntologies(ontList):
    allHypernyms, allConcepts = [], []
    for ont in ontList:
        hypernyms, concepts = extractHypernymsFromOntology(ont)
        allHypernyms.extend(hypernyms)
        allConcepts.extend(concepts)
    return (allHypernyms, allConcepts)

# Enter all ontologies you want parsed
ontologies = os.listdir("ontologies")
hypernyms, words = parseOntologies(ontologies)

concepts = []
for word in words:
    if word.name:
        concepts.append(camel_case_split(word.name))
    else:
        concepts.append(camel_case_split(word.id.strip(":")))

concepts = list(set(concepts))

open("concepts_" + domainName + ".txt","w+").write("\n".join(list(set(concepts))))


# Source 1: DBPedia

def dbpedia_parse(termlist):
    final_list = []
    idx = 0
    for termname in termlist:
        tempList = []
        queryWord = "_".join(termname.split(" "))
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        sparql.setQuery("""SELECT * WHERE {<http://dbpedia.org/resource/"""+queryWord + """> <http://purl.org/linguistics/gold/hypernym> ?hypernyms .}""")
        idx+=1
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query().convert()
        except Exception as e:
            print (idx, queryWord)
            print (e)
            continue

        if results["results"]["bindings"]:
            for result in results["results"]["bindings"]:
                res = result["hypernyms"]["value"]
                name = res.split('/')[-1]
                tempList.append([termname, name, "Hypernym"])
        else:
            termname2 = termname.lower()[0].upper() + termname.lower()[1:]
            queryWord2 = "_".join(termname2.split(" "))
            sparql = SPARQLWrapper("http://dbpedia.org/sparql")
            sparql.setQuery("""SELECT * WHERE {<http://dbpedia.org/resource/"""+queryWord2 + """> <http://purl.org/linguistics/gold/hypernym> ?hypernyms .}""")
            sparql.setReturnFormat(JSON)
            idx+=1
            try:
                results = sparql.query().convert()
            except Exception as e:
                print (idx, queryWord2)
                print (e)
                continue

            for result in results["results"]["bindings"]:
                res = result["hypernyms"]["value"]
                name = res.split('/')[-1]
                tempList.append([termname, name, "Hypernym"])
        
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        sparql.setQuery("""SELECT * WHERE {?hypernyms <http://purl.org/linguistics/gold/hypernym> <http://dbpedia.org/resource/"""+queryWord + """> .}""")
        sparql.setReturnFormat(JSON)
        idx+=1
        try:
#             print (queryWord)
            results = sparql.query().convert()
        except Exception as e:
            print (idx, queryWord)
            print (e)
            continue

        if results["results"]["bindings"]:
            for result in results["results"]["bindings"]:
                res = result["hypernyms"]["value"]
                name = res.split('/')[-1]
                tempList.append([name, termname, "Hyponym"])
        else:
            termname2 = termname.lower()[0].upper() + termname.lower()[1:]
            queryWord2 = "_".join(termname2.split(" "))
            sparql = SPARQLWrapper("http://dbpedia.org/sparql")
            sparql.setQuery("""SELECT * WHERE {?hypernyms <http://purl.org/linguistics/gold/hypernym> <http://dbpedia.org/resource/"""+queryWord2 + """> .}""")
            sparql.setReturnFormat(JSON)
            idx+=1
            try:
#                 print (queryWord2)
                results = sparql.query().convert()
            except Exception as e:
                print (idx, queryWord2)
                print (e)
                continue
            for result in results["results"]["bindings"]:
                res = result["hypernyms"]["value"]
                name = res.split('/')[-1]
                tempList.append([name, termname, "Hyponym"])

        appendingList = []
        for elem in tempList:
            if elem not in appendingList:
                appendingList.append(elem)
        final_list.extend(appendingList)
            
    return final_list

dbpedia_hypernyms_unfiltered = dbpedia_parse(wordsList)

string = "\n".join(["\t".join(a) for a in dbpedia_hypernyms_unfiltered])
open("files/" + domainName + "_dbpedia_unfiltered.txt", "w+").write(string)


# Source 2: Wikidata

def getItems(site, itemtitle):
    params = { 'action' :'wbsearchentities' , 'format' : 'json' , 'language' : 'en', 'type' : 'item', 'search': itemtitle}
    request = api.Request(site=site,**params)
    return request.submit()

# Login to wikidata
def parse_wikidata(termsList):   
    allNames = []
    f = open(domainName + "_wikidata.txt","a+")
    global currentCount
    for term in termsList[currentCount:]:
        currentCount += 1
        site = pywikibot.Site("wikidata", "wikidata")
        repo = site.data_repository()
        wikidataEntries = getItems(site, term)
        # Print the different Wikidata entries to the screen
        # prettyPrint(wikidataEntries)

        # Print each wikidata entry as an object
        for wdEntry in wikidataEntries["search"]:
            entity_id = wdEntry["id"]
            with urllib.request.urlopen("https://www.wikidata.org/w/api.php?action=wbgetentities&ids=" + entity_id + "&format=json") as url:
                data = json.loads(url.read().decode())
                allClaims = data["entities"][entity_id]["claims"]
                if "P31" in allClaims:
                    allInstances = allClaims["P31"]
                    for inst in allInstances:
                        try:
                            currentid = inst["mainsnak"]["datavalue"]["value"]["id"]
                            with urllib.request.urlopen("https://www.wikidata.org/w/api.php?action=wbgetentities&props=labels&ids="+currentid+"&languages=en&format=json") as suburl:
                                dat = json.loads(suburl.read().decode())
                                instanceName = dat["entities"][currentid]["labels"]["en"]["value"]
                            with urllib.request.urlopen("https://www.wikidata.org/w/api.php?action=wbgetentities&props=labels&ids="+entity_id+"&languages=en&format=json") as suburl:
                                dat = json.loads(suburl.read().decode())
                                conceptName = dat["entities"][entity_id]["labels"]["en"]["value"]
                            allNames.append([instanceName, conceptName])
                            string = instanceName + "\t" + conceptName + "\n"
                            f.write(string)
                        except:
                            continue
                if "P279" in allClaims:
                    allInstances = allClaims["P279"]
                    for inst in allInstances:
                        try:
                            currentid = inst["mainsnak"]["datavalue"]["value"]["id"]
                            with urllib.request.urlopen("https://www.wikidata.org/w/api.php?action=wbgetentities&props=labels&ids="+currentid+"&languages=en&format=json") as suburl:
                                s = suburl.read().decode()
                                print (s)
                                dat = json.loads(s)
                                instanceName = dat["entities"][currentid]["labels"]["en"]["value"]
                            with urllib.request.urlopen("https://www.wikidata.org/w/api.php?action=wbgetentities&props=labels&ids="+entity_id+"&languages=en&format=json") as suburl:
                                dat = json.loads(suburl.read().decode())
                                conceptName = dat["entities"][entity_id]["labels"]["en"]["value"]
                            allNames.append([instanceName, conceptName])
                            string = instanceName + "\t" + conceptName + "\n"
                            f.write(string)
                        except:
                            continue

    return allNames


wikidata_hypernyms = parse_wikidata(wordsList)

open("files/" + domainName + "_wikidata.txt","w+").write("\n".join(["\t".join(tup) for tup in wikidata_hypernyms]))

# Source 3: WordNet
def get_relations(termlist):
    final_list = []
    for termname in termlist:
        term = termname.lower()
        print(term)
        for concept in wn.synsets(term):
            for instance in concept.hypernyms():
                instance_name,concept_name = instance.name().split('.')[0], concept.name().split('.')[0]
                # print(instance_name,concept_name)
                final_list.append([instance_name,concept_name])
    return final_list
wordnet_relations = get_relations(concepts)

wordnetString = "\n".join(["\t".join([" ".join(tup[0].split("_")), " ".join(tup[1].split("_"))]) for tup in wordnet_relations])
open("files/" + domainName + "_wordnet.txt","w+").write(wordnetString)

dbpedia = open(domainName + "_dbpedia_unfiltered.txt", "r").read().split("\n")
wikidata = open(domainName + "_wikidata.txt", "r").read().split("\n")
wordnet = open(domainName + "_wordnet.txt", "r").read().split("\n")

allHypernyms = "\n".join(dbpedia + wikidata + wordnet)
open(domainName + "_dataset_untagged.txt", "w+").write(allHypernyms)