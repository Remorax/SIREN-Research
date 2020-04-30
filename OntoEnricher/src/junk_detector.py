import os,shelve
import concurrent.futures

folders = ["../junk/Files/" + l + "/" for l in os.listdir("../junk/Files/") if l.startswith("security_threshold_")]


print ("Len files", len(folders))
idx = 0
def run(f):
    passed1, passed2, passed3, passed4, passed5 = False, False, False, False, False
    print ("Entering for ", f) 
    if len(list(shelve.open(f + "security_relations_map.db", "r").items())):
        passed1 = True
        if len(list(shelve.open(f + "security_path_to_id_dict.db", "r").items())) and len(list(shelve.open(f + "security_id_to_path_dict.db", "r").items())):
            passed2 = True

            s1 = list(shelve.open(f + "security_id_to_word_dict.db", "r").keys())
            s2 = list(shelve.open(f + "security_word_to_id_dict.db", "r").values())
            if len(s1) and len(s2):
                passed3 = True
                s1_sort = sorted(s1, key=lambda l:float(l))
                s2_sort = sorted(s2, key=lambda l:float(l))
                if s1_sort[0] == 0 and s2_sort[0] == 0:
                    passed4 = True
                if s1_sort == s2_sort:
                    passed5 = True
    global idx
    idx += 1
    print (idx, "out of", len(folders),  "Exiting for ", f) 
    return (f, passed1, passed2, passed3, passed4, passed5)

results = []
with concurrent.futures.ProcessPoolExecutor() as executor:
    for res in executor.map(run, folders):
        print (res)
        results.append(res)

files, r1, r2, r3, r4, r5 = list(zip(*results))
r1 = [i for i in zip(files, r1) if i[1]]
r2 = [i for i in zip(files, r2) if i[1]]
r3 = [i for i in zip(files, r3) if i[1]]
r4 = [i for i in zip(files, r4) if i[1]]
r5 = [i for i in zip(files, r5) if i[1]]

print ("Folders having uncorrupted relation dict", r1)
print ("Folders having uncorrupted path-to-id dict", r2)
print ("Folders having uncorrupted word-to-id dict", r3)
print ("Folders having zero-indexed word-to-id dict", r4)
print ("Folders having equal word-to-id and id-to-word dicts", r5)
