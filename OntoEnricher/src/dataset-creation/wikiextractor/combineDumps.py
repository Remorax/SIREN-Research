dirs = ["AA", "AB", "AC"]
import os
allTexts = []
for dir in dirs:
	for filename in os.listdir("text/" + dir):
		text = open("text/" + dir + "/" + filename).read()
		allTexts.append(text)
open("final.txt","w+").write("\n".join(allTexts))