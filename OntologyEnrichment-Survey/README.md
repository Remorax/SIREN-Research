# README

This repository contains the implementations of some of the seminal papers on Ontology Enrichment, namely:
- [Guided Clustering](https://pub.uni-bielefeld.de/download/2497720/2525546/Cimiano_Learning_Concep_1.pdf)
- [C-Pankow](https://dl.acm.org/citation.cfm?id=1060796)
- [Formal Concept Analysis](https://www.jair.org/index.php/jair/article/download/10421/24984)
- [Word2Vec](http://ceur-ws.org/Vol-1690/paper37.pdf)

We tried to enrich the [Information Security](https://gist.github.com/Remorax/f2d6beb580487ffbf29ba9e9cca63f2b) and the [Pizza](https://protege.stanford.edu/ontologies/pizza/pizza.owl) ontologies as input using each of these algorithms. The results, after annotation by a domain expert, can be found [here](https://github.com/Remorax/SIREN-Research/tree/master/OntologyEnrichment-Survey/results). We used these annotated results to calculate the accuracies for each algorithm and proceeded to perform a detailed analysis for each of the algorithms as shown below:

## Clustering

### Accuracy: 
- Pizza: 15.232%
- Security: 13.198%
- Average: 14.215%

### Conclusions:
- It extracts from Web in bulk without looking at page quality or page relevance, both of which give a lot of meaningless output
- Hearst Patterns don't look at the context
- Performs poorly in concepts where there is ambiguity. For example "Four Seasons" and "Rosa", both of which are types of pizza but we mainly get output related to hotels (Four Seasons hotel) for the former and plants (Rose and its types) for the latter
- Performs better in the case of concepts with little ambiguity. For example, "Food" and "Ice Cream". This can be easily explained by the fact that this algorithm does not attempt to perform any disambiguation using context relevance.
- Using Web corpus as input not only results in a lot of junky output, it also prevents us from enriching an ontology with respect to a specific text document of credible quality. And in our test document, searching using Hearst Patterns resulted in 0 matches. 
- The rare occurence of Hearst Patterns in text makes the use of Web as a corpus indispensable, but Web has a lot of low quality content that Guided Clustering does not attempt to filter.
- While it is able to extract concepts and instances that have a relationship, it is NOT able to identify what the relationship is. Not very useful for ontology enrichment.

### Results:
- Pizza: https://docs.google.com/spreadsheets/d/1vluK9HuzENhf0InIvM8tqyX295QuzFtezAkGgjUnOxk/edit?usp=sharing
- Security: https://docs.google.com/spreadsheets/d/1YPmo1A-IlYmNG1zuMoCI3vHvmeqapVTE-Iv8YivXciw/edit?usp=sharing


## C-PANKOW

### Accuracy:
- Pizza: 21.27%
- Security: 20.935% 
- Average: 21.103%

### Conclusions:
- Using Doc2Vec as an indicator for document similarity combined with max vote significantly reduces junk and thus improves accuracy over Guided Clustering
- However accuracy is still very low as both rely on naive pattern-matching without taking context into consideration
- Most of the terms obtained are from WordNet and they are all too generic to be added to an ontology, thus most of them are junk.


### Results
- Pizza: https://docs.google.com/spreadsheets/d/1aD7ASgPW0LC0GMzM06Gy9MlKqsG4FCowit870HaHhhw/edit?usp=sharing
- Security: https://docs.google.com/spreadsheets/d/11lcB04y5M6eK2SN8oCl1QXj1_TCUG_809eKIRm-XCuI/edit?usp=sharing


## FCA

### Accuracy:
- Pizza: 22.727%
- Security: 21.66%
- Average: 22.19%

### Conclusions:
- Uses SVO triplet extraction which is an improvement on the pattern-based matching so it results in a slight increase in performance
- However again, like rest of the path-based algorithms, it does not take context into consideration.
- Results in a lot of junk output unrelated to the domain
- Lack of coreference resolution results in a lot of pronouns being extracted as well
- Also, using a verb as a parent for grouping concepts of a specific sentence is not always the best idea as the concept may be embedded in phrases, sentences or a group of sentences.

### Results:
- Pizza: https://docs.google.com/spreadsheets/d/1z3j2vrJ9F7jhN2gi2oLjau3eqyiIrA5JCFup-SH9oCM/edit?usp=sharing
- Security: https://docs.google.com/spreadsheets/d/1IHVzAC-9kAAGk4k7VxqF1Yu--ZNEUQUqf3BLKc6xuUg/edit?usp=sharing


## Word2Vec

### Accuracy:
- Pizza: 50%
- Security: 37.73%
- Average: 43.865%

### Conclusions:
- Performs significantly better compared to path-based approaches as it uses the word2vec model trained on Google News corpus to find the most similar words related to the ones in the seed ontology. This shows even the most basic and naive distributional approaches outperform pattern-based approaches proposed through the 2000s
- Requires manual intervention at every stage
- Again, doesn't extract concepts from a text file. Extracts from the huge text corpus (Google News corpus) the Word2Vec model was trained on, which needs to be very large (~10GB) in order for the model to learn representations appropriately.
- However training the model on very large corpora can compromise on the domain specific knowledge. Moreover, with increasing amounts of data generated on the internet on a daily basis, ontologies often need to be enriched with the concepts from a single test document, not a specific huge corpus, as in Word2Vec or the unfiltered Web, as in Guided Clustering or C-PANKOW.
- Also detects only is-a relationships. A huge limitation.
- Also, empirically we found that most relationships identified as "is-a" aren't actually "is-a". They reflect synonymy or some other relationship that doesn't fall under "is-a" relationships.
- Only 1 of the relationships identified had one of its terms as one of the concepts in the seed ontology, therefore only 1 of the identified relationships could be added to the original ontology

### Results:
- Pizza: https://docs.google.com/spreadsheets/d/1sCjEeloRFnaP3PReYNZ4dYuz76T4T8dqawPMKyelPuw/edit?usp=sharing
- Security: https://docs.google.com/spreadsheets/d/1jwBKZuIrgwb31b0XeBrvlZSl2YOQ_UyaBoNU-ZE1i-A/edit?usp=sharing
