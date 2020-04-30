import os,shelve
import concurrent.futures

folders = ["../junk/Files/" + l + "/" for l in os.listdir("../junk/Files/") if l.startswith("security_threshold_")]

passed1, passed2, passed3, passed4, passed5 = [], [], [], [], []
print ("Len files", len(folders))
idx = 0
def run(f):
    print ("Entering for ", f) 
    if len(list(shelve.open(f + "security_relations_map.db", "r").items())):
        passed1.append(f)
        if len(list(shelve.open(f + "security_path_to_id_dict.db", "r").items())) and len(list(shelve.open(f + "security_id_to_path_dict.db", "r").items())):
            passed2.append(f)

            s1 = list(shelve.open(f + "security_id_to_word_dict.db", "r").keys())
            s2 = list(shelve.open(f + "security_word_to_id_dict.db", "r").values())
            if len(s1) and len(s2):
                passed3.append(f)
                s1_sort = sorted(s1, key=lambda l:float(l))
                s2_sort = sorted(s2, key=lambda l:float(l))
                if s1_sort[0] == 0 and s2_sort[0] == 0:
                    passed4.append(f)
                if s1_sort == s2_sort:
                    passed5.append(f)
    global idx
    idx += 1
    print (idx, "out of", len(folders),  "Exiting for ", f) 


with concurrent.futures.ProcessPoolExecutor() as executor:
    for res in executor.map(run, folders):
        pass
print ("Folders having uncorrupted relation dict", passed1)
print ("Folders having uncorrupted path-to-id dict", passed2)
print ("Folders having uncorrupted word-to-id dict", passed3)
print ("Folders having zero-indexed word-to-id dict", passed4)
print ("Folders having equal word-to-id and id-to-word dicts", passed5)
