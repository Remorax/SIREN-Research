import subprocess
from os import listdir
from os.path import isfile, join
import xml.dom.minidom
from owlready2 import *
from onto_app import db
from onto_app.aggregate import accepted
from rdflib import Graph
from rdflib.namespace import OWL, RDF, RDFS
from collections import defaultdict

OWL2VOWL = 'OWL2VOWL-0.3.5-shaded.jar'
baseurl = "https://serc.iiit.ac.in/downloads/ontology/test.owl"

def is_blank(node):
    if not '#' in node:
        return True
    else:
        return False

def run(inputfile, outputfile, url, results):
    print (outputfile)
    thefile=open(outputfile,"w+")
    doc = xml.dom.minidom.parse(inputfile)
    print ("Results",results)
    for (instance,relation,concept) in results:
        # needed = doc.getElementsByTagName("owl:Class")
        
        # (instance, concept) = line.split()
        # needed = doc.getElementsByTagName("owl:Class")
        # newelementclass = doc.createElement("owl:Class")
        # newelementclass.setAttribute("rdf:about", url + "#" + concept)
        # newelementsubclass = doc.createElement("rdfs:subClassOf")
        # newelementsubclass.setAttribute("rdf:resource",url + "#" + instance)
        # newelementclasslabel = doc.createElement("rdfs:label")
        # newelementclasslabel.setAttribute("xml:lang","en")
        # text = doc.createTextNode(concept)
        # newelementclasslabel.appendChild(text)
        # newelementclass.appendChild(newelementsubclass)
        # newelementclass.appendChild(newelementclasslabel)
        # needed[0].parentNode.insertBefore(newelementclass, needed[0])
        # 
        
        # if search.getAttribute("rdf:parseType"):
        #     newelementclass = doc.createElement("rdf:Description")
        #     newelementclass.setAttribute("rdf:about", url + "#" + concept)
        #     search.appendChild(newelementclass)

            
        #     newelementclass = doc.createElement("rdf:Description")
        #     newelementclass.setAttribute("rdf:about", url + "#" + instance)
        #     search.appendChild(newelementclass)
        x = doc.getElementsByTagName(baseurl + "#" + relation)
        if not x:
            # newRelationshipClass = doc.createElement(url + "#" + relation)
            newElementObject = doc.createElement("owl:ObjectProperty")
            newElementObject.setAttribute("rdf:about", url + "#" + relation)
            newTypeObject = doc.createElement("rdf:type")
            newTypeObject.setAttribute("rdf:resource","http://www.w3.org/2002/07/owl#FunctionalProperty")
            newElementObject.appendChild(newTypeObject)
            # newRelationshipClass.appendChild(newElementObject)
            doc.childNodes[0].appendChild(newElementObject)
        else:
            pass
        needed = doc.getElementsByTagName("owl:Class")
        newElementClass = doc.createElement("owl:Class")
        newElementClass.setAttribute("rdf:about", url + "#" + concept)
        newElementSubClass = doc.createElement("rdfs:subClassOf")
        newElementRestriction = doc.createElement("owl:Restriction")
        newElementProperty = doc.createElement("owl:onProperty")
        newElementProperty.setAttribute("rdf:resource", url + "#" + relation)
        newElementsomeValuesFrom = doc.createElement("owl:someValuesFrom")
        newElementsomeValuesFrom.setAttribute("rdf:resource", url + "#" + instance)
        newelementclasslabel = doc.createElement("rdfs:label")
        newelementclasslabel.setAttribute("xml:lang","en")
        text = doc.createTextNode(concept)
        newelementclasslabel.appendChild(text)
        newElementRestriction.appendChild(newElementProperty)
        newElementRestriction.appendChild(newElementsomeValuesFrom)
        newElementSubClass.appendChild(newElementRestriction)
        newElementClass.appendChild(newElementSubClass)
        newElementClass.appendChild(newelementclasslabel)
        needed[0].parentNode.insertBefore(newElementClass, needed[0])
        search = doc.getElementsByTagName("owl:members")[0]
        if search.getAttribute("rdf:parseType"):
            newelementclass = doc.createElement("rdf:Description")
            newelementclass.setAttribute("rdf:about", url + "#" + concept)
            search.appendChild(newelementclass)

            
            newelementclass = doc.createElement("rdf:Description")
            newelementclass.setAttribute("rdf:about", url + "#" + instance)
            search.appendChild(newelementclass)
        # x = doc.getElementsByTagName()
        # needed = doc.getElementsByTagName("owl:Class")
        # newelementclass = doc.createElement("owl:Class")
        # newelementclass.setAttribute("rdf:about", url + "#" + concept)
        # # newelementsubclass = doc.createElement("rdfs:subClassof")
        # # newelementsubclass = doc.createElement("rdfs:" + relation)
        # newelementsubclass.setAttribute("rdf:resource",url + "#" + instance)
        # newelementclasslabel = doc.createElement("rdfs:label")
        # newelementclasslabel.setAttribute("xml:lang","en")
        # text = doc.createTextNode(concept)
        # newelementclasslabel.appendChild(text)
        # newelementclass.appendChild(newelementsubclass)
        # newelementclass.appendChild(newelementclasslabel)
        # needed[0].parentNode.insertBefore(newelementclass, needed[0])
        # search = doc.getElementsByTagName("owl:members")[0]
    

        
    doc.writexml(thefile)
    thefile.close()
    print("Done")

def createParsedRelations(file, fname):
    allParsedRelations = []
    for line in open(file, "r").readlines():
        if line.split():
            (instance, relation, concept) = line.split()
            newinstance = baseurl + "#" + concept 
            newconcept = baseurl + "#" + instance
            relation = baseurl+relation
            allParsedRelations.append(" ".join([newinstance, relation, newconcept]))
    string = "\n".join(allParsedRelations)
    open("./data/new/" + str(fname) + '.txt', "w+").write(string)
    return


def add_onto_file(admin_id, name):
    # compile OWL to JSON using OWL2VOWL
    json_path = './data/json/' + str(name) + '.json'
    unparsed_relations_file = './data/input/' + str(name) + '.txt'
    filepath = './data/input/' + str(name) + '.owl'
    f = open(json_path, 'w')
    allTriples = [el.strip(" ").split(" ") for el in open(unparsed_relations_file).read().split("\n") if el]
    print(allTriples)
    createParsedRelations(unparsed_relations_file, name)
    new_relations_file = './data/new/' + str(name) + '.txt'
    outputfile = "./data/owl/" +str(name) + '.owl'
    print ("Hi im here")
    run(filepath,outputfile,baseurl, allTriples)
    try:
        subprocess.run(['java', '-jar', OWL2VOWL, '-file', outputfile, '-echo'], stdout=f)
    except:
        raise RuntimeError

    # Create record for ontology in database
    insert_query = """INSERT INTO ontologies (name, admin_id)
                        VALUES (:name, :admin_id)"""
    result = db.engine.execute(insert_query, {'name': str(name), 'admin_id': admin_id})#'filepath': filepath, )
    new_ontology_id = result.lastrowid
    db.session.commit()

    # add new relations to database
    new_relations,new_subclasses, new_nodes = get_new_relations(new_relations_file,unparsed_relations_file)
    add_relations_to_db(new_relations, new_ontology_id)
    add_nodes_to_db(new_nodes, new_ontology_id)
    # add_subclasses_to_db(new_subclasses, new_ontology_id)

def add_new_ontologies():
    ontologies = ['.'.join(f.split('.')[:-1]) for f in listdir("./data/owl/") if isfile(join("./data/owl/", f))]
    ontologies = [ont for ont in ontologies if ont]
    print("Onto=", ontologies)
    result = db.engine.execute("""SELECT name FROM ontologies""")
    db_ontologies = [o['name'] for o in result.fetchall()]
    for onto in ontologies:
        if not (onto in db_ontologies):
            add_onto_file(0, onto)

def get_new_relations(filepath,txtfile_path):
    d = dict()
    print (filepath)
    f = open(txtfile_path, 'r')
    relations = list()
    classes = list()
    subclasses = list()
    print(filepath)
    onto = get_ontology(filepath).load()
    existing_nodes = []
    nodes = []
    print(list(onto.classes()))
    for c in list(onto.classes()):
        existing_nodes.append(str(c._name))
    for line in f.readlines():
         if line.split():
            (instance, relation, concept) = line.split()
            nodes.append(instance)
            nodes.append(concept)
            newinstance = baseurl + "#" + concept 
            newconcept = baseurl + "#" + instance
            relation = baseurl+"#"+relation
            relations.append([newinstance, relation, newconcept])
    nodes = set(nodes)
    existing_nodes = set(existing_nodes)
    new_nodes = nodes.difference(existing_nodes)
    print(new_nodes)
    print(nodes.difference(existing_nodes))
    # relations = set(relations)
    new_nodes = list(new_nodes)
    print(nodes)
    print(existing_nodes)
    print(new_nodes)
    print(relations)
    n_nodes = []
    for concept in new_nodes:
        new_concept = baseurl + "#" + concept
        n_nodes.append(new_concept) 
    return relations, [],n_nodes

    
    # Each line of the new relations file is an RDF triple, so it is a
    # triple of the subject, predicate, and object
    # Create an adjacency list graph from the triples
    # for l in f.readlines():
    #     print("new_file",l)
    #     print("\n")
    #     s, p, o = l.split()
       

    #     if o == str(OWL.Class):
    #         classes.append(s)
    #     elif (p == str(RDFS.subClassOf) and not is_blank(s) and not is_blank(o)):
    #         subclasses.append((s, o))
    #     else:
    #         if s in d:
    #             d[s].append((p, o))
    #         else:
    #             d[s] = [(p, o)]

    # # From the graph, find all restricitons (blank nodes) and get the relevant
    # # relation data from them
    # for s in d:
    #     if not is_blank(s):
    #         for p, o in d[s]:
    #             if not is_blank(o):
    #                 domain = s
    #                 rang = None
    #                 quant = None
    #                 prop = None
    #                 for p1, o1 in d[o]:
    #                     if p1 == str(OWL.onProperty):
    #                         prop = o1
    #                     elif p1 == str(OWL.someValuesFrom):
    #                         quant = p1
    #                         rang = o1
    #                 if quant == str(OWL.someValuesFrom):
    #                     relations.append((domain, prop, quant, rang))
    # print(subclasses, relations, classes)
    # return relations, classes, subclasses

 
def add_nodes_to_db(nodes, onto_id):
    insert_query = """INSERT INTO
                    nodes (name, onto_id)
                    VALUES (:name, :onto_id)"""
    args = {'name': None, 'onto_id': onto_id}
    for n in nodes:
        args['name'] = n
        result = db.engine.execute(insert_query, args)
        # print(result)
    # db.session.commit()

def add_relations_to_db(relations, onto_id):
    insert_query = """INSERT INTO
                    class_relations (domain, property, quantifier, range, onto_id)
                    VALUES (:domain, :property, :quantifier, :range, :onto_id)"""
    args = {'domain': None, 'property': None, 'quantifier': None, 'range': None, 'onto_id': onto_id}
    print("#relations = ", relations)
    for r in relations:
        args['domain'] = r[0]
        args['property'] = r[1]
        args['quantifier'] = None
        args['range'] = r[2]
        result = db.engine.execute(insert_query, args)
        print(result)
    # db.session.commit()

def add_subclasses_to_db(subclasses, onto_id):
    insert_query = """INSERT INTO
                    class_relations (domain, property, quantifier, range, onto_id)
                    VALUES (:domain, :property, :quantifier, :range, :onto_id)"""
    args = {'domain': None, 'property': None, 'quantifier': None, 'range': None, 'onto_id': onto_id}
    # print("#relations = ", len(relations))
    for r in subclasses:
        print (r)
        args['domain'] = r[0]
        args['property'] = None
        args['quantifier'] = str(RDFS.subClassOf)
        args['range'] = r[1]
        result = db.engine.execute(insert_query, args)
        print(result)
    # db.session.commit()


def add_relation_with_credibility_only(twitter_users):
    # query = """SELECT * FROM class_decisions"""
    # result = db.engine.execute(query)
    query = """SELECT * FROM class_decisions INNER JOIN class_relations ON class_decisions.relation_id =class_relations.id where class_decisions.id= ?"""
    result = db.engine.execute(query)
    relation_list = [(o['relation_id'],o['property'],o['domain'],o['range']) for o in result.fetchall()]
    relation_set = set(relation_list)
    relation_dict = defaultdict(int)
    relation_count = defaultdict(int)
    query = """SELECT * FROM class_decisions INNER JOIN class_relations ON class_decisions.relation_id =class_relations.id where class_decisions.id= ?"""
    result = db.engine.execute(query)
    for tup in relation_set :
        for o in result.fetchall():
            if(tup[0] == o['property'] and tup[1] == o['domain'] and tup[2] == o['range']):
                relation_dict[tup]+=(o['approved']*twitter_users[o['user_id']])
                relation_count[tup]+=twitter_users[o['user_id']]
            else:
                pass


    query = """DELETE * FROM class_decisions WHERE class_decisions.id = ?"""
    result = db.engine.execute(query)
    for tup,score in relation_dict.items():
        score = score/relation_count[tup]
        relation_dict[tup] = score
    for tup,score in relation_dict.items():
        if score > 0.5:
            args = {
                    'relation_id': tup[0],
                        # 'property': property,
                    'approved': 1,
                    'user_id': None
                    }
            insert_query = """INSERT INTO class_decisions
                        (relation_id, user_id, approved)
                        VALUES (:relation_id, :user_id, :approved)"""
            db.engine.execute(insert_query,args)
        else:
            args = {
                    'relation_id': tup[0],
                        # 'property': property,
                    'approved': 0,
                    'user_id': None
                    }
            insert_query = """INSERT INTO class_decisions
                        (relation_id, user_id, approved)
                        VALUES (:relation_id, :user_id, :approved)"""
            db.engine.execute(insert_query,args)








def add_relation_decision(user_id, property, domain, range, quantifier, onto_id, decision):
    args = {
        'onto_id': onto_id,
        # 'property': property,
        'domain': domain,
        'range': range
        # 'quantifier': quantifier
    }
    print("user_id:", user_id)
    print("user_id:", property)
    print("domain:", domain)
    print("range:", range)
    print("quantifier:", quantifier)
    print("onto_id:", onto_id)
    print("decision:", decision)

    if property:
        args['property'] = property
        relation_query = """SELECT id FROM class_relations
                        WHERE onto_id = :onto_id
                            AND property = :property
                            AND domain = :domain
                            AND range = :range"""
    else:
        relation_query = """SELECT id FROM class_relations
                        WHERE onto_id = :onto_id
                            AND domain = :domain
                            AND range = :range"""

    result = db.engine.execute(relation_query, args)


    relation_id = result.fetchone()['id']

    insert_query = """INSERT INTO class_decisions
                        (relation_id, user_id, approved)
                        VALUES (:relation_id, :user_id, :approved)"""
    print (relation_id, user_id, decision)
    with db.engine.connect() as connection:
        result = connection.execute(insert_query, {
            'relation_id': relation_id,
            'user_id': user_id,
            'approved': decision
        })

def add_node_decision(user_id, name, onto_id, decision):
    relation_query = """SELECT id FROM nodes
                        WHERE onto_id = :onto_id
                            AND name = :name"""

    result = db.engine.execute(relation_query, {
        'onto_id': onto_id,
        'name': name,
    })
    print(user_id)
    print(onto_id)
    print(decision)
    print(name)
    node_id = result.fetchone()['id']

    result = db.engine.execute("""SELECT * FROM node_decisions 
            WHERE user_id = :user_id AND node_id = :node_id""", {'user_id': user_id, 'node_id': node_id})
    
    if result.fetchone():
        db.engine.execute("""UPDATE node_decisions SET approved = :decision
        WHERE user_id = :user_id AND node_id = :node_id""", 
        {'user_id': user_id, 'node_id': node_id, 'approved': decision})
    else:
        insert_query = """INSERT INTO node_decisions
                            (node_id, user_id, approved)
                            VALUES (:node_id, :user_id, :approved)"""
        result = db.engine.execute(insert_query, {
            'node_id': node_id,
            'user_id': user_id,
            'approved': decision
        })

def get_decision(relation_id):
    query = """SELECT * FROM class_decisions WHERE relation_id = :relation_id"""
    result = db.engine.execute(query, {'relation_id': relation_id})
    return accepted(result.fetchall())

def get_ontologies_on_server():
    ontologies = ['.'.join(f.split('.')[:-1]) for f in listdir("./data/owl/") if isfile(join("./data/owl/", f)) and f.endswith(".owl")]
    print(ontologies)
    return ontologies
