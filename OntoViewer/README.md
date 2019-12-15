# Ontology Determiner
# ONTOLOGY VIEWER

# Requirements:
- python3 and pip are prerequisites

# SetUp:
- go to src folder and run `pip install -r requirements.txt`, incase of any missing modules , please report

# Data Requirements:
Place your seed file in src/onto_app/data/input as .owl file
Place your new relations and concepts file in src/onto_app/data/input as .txt file
The name of both .owl file and .txt file should be the same 
Say for example pizza.owl and pizza.txt


# Ontology Viewer
- go to 'onto_app' directory int terminal and set `FLASK_APP=routes.py` and `FLASK_ENV=development`
- start the server by flask run
- login to the application by your twitter account 
- accept or reject relationships and concepts
- go back to the login  page and logout
- stop the server

# Final Ontology file
-remove the files in /src/data/final if nay
- run python3 get_verified_ontology.py <file_name>
- final ontology is stored in src/data/final as <file_name>final.owl

# Changing the Source code of WEBVOWL
To see how to change the source code of WEBVOWL go to src/WEBVOWL and read the readme and follow the instructions.
License for using the source code of WEBVOWL is given by MIT and the license file is in src/WEBVOWL




