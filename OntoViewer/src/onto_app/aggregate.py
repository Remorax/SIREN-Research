from sqlalchemy import create_engine

import subprocess, os
from rdflib.namespace import OWL, RDF, RDFS
import xml.dom.minidom
RESTRICTIONS = "Restriction_removal-1.0-SNAPSHOT-jar-with-dependencies.jar"
CLASSES = "Class_removal-1.0-SNAPSHOT-jar-with-dependencies.jar"
SUBCLASSES = "Subclass_removal-1.0-SNAPSHOT-jar-with-dependencies.jar"
base_url = "https://serc.iiit.ac.in/downloads/ontology/test.owl"
def accepted(decisions):
    count_accept = 0
    count_reject = 0
    for d in decisions:
        if d == 0:
            count_reject += 1
        else:
            count_accept += 1
    return bool(count_accept >= count_reject)

def generate_final_ontology(name):
    engine = create_engine('sqlite:///onto.db', echo = True)
    # conn = sqlite3.connect('onto.db')
    # c = conn.cursor()
    c = engine.connect()
    trans = c.begin()
    try:
        result = c.execute('''SELECT id FROM ontologies WHERE name = ?''', (name,))
        onto_id = None
        res = result.fetchone()
        if res:
            onto_id = res[0]
        else:
            raise ValueError
        print ("ontoId:",onto_id)
        owl_path = './data/final/' + name + '.owl'
        if not os.path.isfile(owl_path):
            try:
                subprocess.run(['cp', './data/owl/' + name + '.owl', owl_path])
            except:
                raise RuntimeError
        
        result = c.execute("""SELECT * FROM class_relations WHERE onto_id = ?""", (onto_id,))
        relations = result.fetchall()
        print ("**",relations, "**")
        for r in relations:
            # print (r)
            result = c.execute("""SELECT * FROM class_decisions WHERE relation_id = ?""", (r[0],))
            decisions = result.fetchall()
            print("Yoda")
            if decisions:
                print (type(decisions), type(decisions[0]))
                print("Double YOda")
                if not accepted([d['approved'] for d in decisions]):
                    print(type(r['quantifier']))
                    print(type(RDFS.subClassOf))
                    print("Triple Yoda")
                    if r['quantifier'] == str(RDFS.subClassOf):
                        try:
                            subprocess.run(['java', '-jar', SUBCLASSES, r['domain'], r['range'], owl_path])
                        except:
                            raise RuntimeError
                    else:
                        try:
                            print("Quadrauple Yoda")
                            subprocess.run(['java', '-jar', RESTRICTIONS, r['domain'], r['property'], r['range'], owl_path])
                        except:
                            raise RuntimeError
                c.execute("""DELETE FROM class_relations WHERE id = ?""", r['id'])
        
        trans.commit()

        # print ("Results",results)
    #     result = c.execute("""SELECT * FROM nodes WHERE onto_id = ?""", (onto_id,))
    #     nodes = result.fetchall()
    #     for n in nodes:
    #         print (n)
    #         result = c.execute("""SELECT * FROM node_decisions WHERE node_id = ?""", n['id'])
    #         decisions = result.fetchall()
    #         print(decisions)
    #         if decisions:
    #             if not accepted([d['approved'] for d in decisions]):
    #                 print("XYZ",n['name'])
    #                 try:
    #                     subprocess.run(['java', '-jar', CLASSES, n['name'], owl_path])
    #                 except:
    #                     raise RuntimeError
    #             c.execute("""DELETE FROM nodes WHERE id = ?""", n['id'])
        
    #     trans.commit()
    except:
        trans.rollback()
        raise

    # print("Done")
    




def delete_nodes(name):
    engine = create_engine('sqlite:///onto.db', echo = True)
    # conn = sqlite3.connect('onto.db')
    # c = conn.cursor()
    c = engine.connect()
    trans = c.begin()
    result = c.execute('''SELECT id FROM ontologies WHERE name = ?''', (name,))
    onto_id = None
    res = result.fetchone()
    if res:
        onto_id = res[0]
    else:
        raise ValueError
    result = c.execute("""SELECT * FROM nodes WHERE onto_id = ?""", (onto_id,))
    nodes = result.fetchall()
    print (nodes)
    print("/home/harish/Documents/onto/Ontology/OntologyDeterminer/Ontology-Determiner/onto_viewer/src/onto_app/data/final/" + name + '.owl')
    thefile=open("./data/final/" + name + '.owl',"r")
    #print(thefile.readlines())
    # for line in thefile.readlines():
    #     print(line)
    outputfile=open("./data/final/pizzafinal.owl","w+")
    doc = xml.dom.minidom.parse(thefile)
    for n in nodes:
        print(n)
        result = c.execute("""SELECT * FROM node_decisions WHERE node_id = ?""", n['id'])
        decisions = result.fetchall()
        print(decisions)
        if decisions:
            print("XYZ",n['name'])
            nod = doc.getElementsByTagName("owl:Class")
            for p in nod:
                print(p.getAttribute("rdf:about"))
                if(p.getAttribute("rdf:about") == n['name']):
                    print(n['name'])
                    p.parentNode.removeChild(p)
            nod = doc.getElementsByTagName("rdf:Description")
            for p in nod:
                if(p.getAttribute("rdf:about")==n['name']):
                    print("should be success")
                    p.parentNode.removeChild(p)
    doc.writexml(outputfile)
    outputfile.close()
    thefile.close()
    trans.commit()


    
    
    