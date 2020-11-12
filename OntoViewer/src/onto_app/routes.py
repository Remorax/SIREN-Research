from flask import Flask, request, redirect, url_for, session, g, flash, \
    render_template
from onto_app import app, db
import os
from requests_oauthlib import OAuth1Session
from requests_oauthlib import OAuth1
from flask import send_file, send_from_directory, redirect, url_for, flash, current_app, session
from werkzeug.utils import secure_filename
import json
from onto_app.onto import *
import tweepy
request_token_url = 'https://api.twitter.com/oauth/request_token'
# app = Flask(__name__)
# Load our config from an object, or module (config.py)

# These config variables come from 'config.py'
client_key = "9NDG7eIVsrouj4CS2M7LoNjM1"
client_secret = 'y1z075l563BwcL8XtI7GzQzEnvo1jEEzmcmR1NFBxhYPFokYzu'
# auth = tweepy.OAuthHandler("9NDG7eIVsrouj4CS2M7LoNjM1",
#                            'y1z075l563BwcL8XtI7GzQzEnvo1jEEzmcmR1NFBxhYPFokYzu')
# auth.set_access_token('1192925360851013632-9tVq9NfbXX1BM1q8pUxMqA3K6ZGIqD',
#                       'CL9MFVQDYUNM3cVuNeg0HAcSFKA4YRER6YKaKKKxNlYeG')
# oauth = OAuth1Session(client_key, client_secret=client_secret)
# fetch_response = oauth.fetch_request_token(request_token_url)
# resource_owner_key = fetch_response.get('oauth_token')
# resource_owner_secret = fetch_response.get('oauth_token_secret')
# base_authorization_url = 'https://api.twitter.com/oauth/authorize'
# tweepy_api = tweepy.API(oauth)
DEBUG = True
SECRET_KEY = 'AbYzXSaNdErS123@'
app.debug = DEBUG
app.secret_key = SECRET_KEY
VERIFIER = "epsteindidntkillhimself"
# This variable specifies the name of a file that contains the OAuth 2.0
# information for this application, including its client_id and client_secret.
CLIENT_SECRETS_FILE = "client_secret_395200844618-bnei4qvc8203ieoic6hpkbrkdnvmdq49.apps.googleusercontent.com.json"
CLIENT_ID = "395200844618-bnei4qvc8203ieoic6hpkbrkdnvmdq49.apps.googleusercontent.com"
tweepy_api = None
# This OAuth 2.0 access scope allows for full read/write access to the
# authenticated user's account and requires requests to use an SSL connection.
SCOPES = ["https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile", "openid"]
API_SERVICE_NAME = 'drive'
API_VERSION = 'v2'
db.init_app(app) 
# prevent cached responses
@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "-1"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/')
def home():
   return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    oauth = tweepy.OAuthHandler(client_key,client_secret)
    url = oauth.get_authorization_url()
    session['request_token'] = oauth.request_token
    # verifier = requests.get('oauth_verifier')
    
    # key = auth.access_token
    # secret = auth.access_token_secret
    # oauth_response = oauth.parse_authorization_response("http://127.0.0.1:5000/authenticated")
    # verifier = oauth_response.get('oauth_verifier')
    
    return redirect(url)   

""" Loads ontology to database """
@app.route('/hello', methods=["GET", "POST"])
def hello():
    if request.method == 'GET' :
        add_onto_file(1, "pizza")
        return "Pizza ontology has been added to database"
    if request.method == 'POST' :
        """ Remanants of testing code, will be removed later when they will be no longer be used with certainity. """
        """ Returning name and type of links. Does not update with objects. Bias for new yet to be set, will be done so when more of the backend for it is built. """
        a = str(request.data).split(',')
        Prop = a[0]
        Type = a[1]
        Decision  = a[2]
        Domain = a[3]
        Range = a[4]

        print(Decision[12:-3])

        i = 0
        try:
            while Prop[i] != '>':
                i += 1
        except:
            print(i)

        if Prop[-5 :-1] == '</a>' :
            print(Prop[i+1:-5])
        else  :
            print(Prop[i+1:-8])

        print(Type[8 : -1])
        """ End of preliminary return of accept return. """

    return render_template("index.html")

@app.route('/authenticated')
def authenticated():
    # Specify the state when creating the flow in the callback so that it can
    # verified in the authorization server response.
    # state = session['state']
    verification = request.args["oauth_verifier"]
    auth = tweepy.OAuthHandler(client_key, client_secret)
    try:
        auth.request_token = session["request_token"]
    except KeyError:
        flash("Please login again", "danger")
        return redirect('login')

    try:
        auth.get_access_token(verification)
    except tweepy.TweepError:
        flash("Failed to get access token", "danger")
        return redirect('login')

    session["access_token"] = auth.access_token
    session["access_token_secret"] = auth.access_token_secret
    tweepy_api = tweepy.API(auth)
    user_object = tweepy_api.me()
    userid = user_object.id
    session['credentials'] = credentials_to_dict(user_object)
    user_name = user_object.screen_name
    # print("Hello", userid, email)
    result = db.engine.execute("SELECT * FROM users WHERE id = :id", {'id': userid})
    if not result.fetchone():
        db.engine.execute("""INSERT INTO users (id, username, privilege) VALUES
                            (:id, :username, :privilege)""", {'id': userid, 'username': user_name, 'privilege': 0})
    session['userid'] = userid
    session['username'] = user_name

    return redirect(url_for('user'))

def credentials_to_dict(credentials):
  return {'id': credentials.id,
          'name': credentials.screen_name,
          }



@app.route('/user')
def user():
    if not 'credentials' in session:
        return redirect(url_for('home'))

    ontologies = get_ontologies_on_server()

    try:
        add_onto_file(1, "pizza")
    except:
        pass
    # return redirect(url_for('loadOntology', filename='pizza.json'))
    return render_template("ontologies.html", ontologies=ontologies, username=session['username'])

@app.route('/logout')
def logout():
    if 'credentials' in session:
        del session['credentials']
        del session['username']
        del session['userid']
    return redirect(url_for('home'))

""" Stores decisions taken in frontend corresponding to relationships accept/reject into database """
@app.route('/decision', methods=["POST"])
def decision() :
    if request.method == 'POST' :
        """ Decisions stored """
        """ Index numbers used to extract specific content from already existing inner html. This will hold through across cases."""
        data = str(request.data).split(',')
        # if flag is 1, then relation, else node
        user_id = session['userid']
        onto_id = session['ontology']
        
        if data[0][-1] == "1" :
            #when a relationship is accepted/rejected
            Prop = data[1][8:-1]
            Type = data[5][8:-1]
            Decision  = data[2][12:-1]
            Domain = data[3][10:-1]
            Range = data[4][9:-1]

            print("Prop : ", Prop)
            print("Domain : ", Domain)
            print("Range : ", Range)
            print("Decision : ", Decision)
            print("Type : ", Type)

            """ Call add_decision from onto.py to store decision """
            if Prop == "Subclass of" :
                add_relation_decision(user_id, None, Domain, Range, str(RDFS.subClassOf), onto_id, {'Accept': 1, 'Reject':0}[Decision] )
            else :
                add_relation_decision(user_id, Prop, Domain, Range, Type, onto_id, {'Accept': 1, 'Reject':0}[Decision])

        elif data[0][-1] == "0" :
            # When a node is accpeted or rejected.
            name = data[1][8:-1]
            Decision = data[2][12:-1]

            # print("Name : ", Name)
            # print("Decision :", Decision)

            """ Call add_decision on node from onto.py to store decision """
            add_node_decision(user_id, name, onto_id, {'Accept': 1, 'Reject':0}[Decision])

    return render_template("index.html")


""" Serve file and new relationships from backend corresponding to the filename given in the URL """
@app.route("/loadOntology/<path:file>/", methods = ['GET'])
def loadOntology(file) :
    """ Serve files and new relations from the backend """
    """ Ontologies ready to be rendered saved in data/json """

    if 'credentials' not in session:
        return redirect('login')

    filename = file + '.json'
    uploads = os.path.join(current_app.root_path,"data/json")
    uploads = uploads + "/" + str(filename)
    print(uploads)
    fname = str(filename)
    fname = fname.split(".")[0]
    fname2 = fname + ".owl"
    fname = fname + ".txt"

    result = db.engine.execute("SELECT id FROM ontologies WHERE name = :name", {'name': file})
    onto_id = result.fetchone()['id']
    session['ontology'] = onto_id
    """ Corresponding new relations for given ontology are stored in data/new. """

    new_relations, new_classes,new_nodes = get_new_relations(os.path.join(current_app.root_path,"data/input")+ "/" + fname2,os.path.join(current_app.root_path,"data/input")+ "/" + fname)
   
    print("new_nodes",new_nodes)
    result = db.engine.execute("""SELECT * FROM class_relations WHERE quantifier != :subclass""",
        {'subclass': str(RDFS.subClassOf)})
    # new_relations = [(r['domain'], r['property'], r['quantifier'], r['range']) for r in result.fetchall()]
    print("new_relations",new_relations)
    result = db.engine.execute("""SELECT * FROM nodes""")
    # /new_nodes = [n['name'] for n in result.fetchall()]
   
    result = db.engine.execute("""SELECT * FROM class_relations WHERE quantifier = :subclass""",
        {'subclass': str(RDFS.subClassOf)})
    new_subclasses = [(r['domain'], r['range']) for r in result.fetchall()]
    print ("new subclass", new_subclasses)
    # nodes = []
    # for i in range(len(new_nodes)):
    #     for j in range(len(new_nodes[i])):
    #         nodes.append(new_nodes[i][j])
    # print("new_nodes",nodes)
    try :
        with open(uploads,"r") as json_data:
            contents = json.load(json_data)
            # print(contents)
    except :
        flash('Oops record not found')
        return redirect(url_for('hello'))
    # new_relations = list(set(new_relations))
    return render_template("index.html", OntologyContentJson=contents,userId=session['userid'], hiddenJSONRel =new_relations, hiddenJSONNode = new_nodes, emptyList = [])


# @app.route('/return-files/<path:filename>/', methods = ['GET', 'POST'])
# def return_files(filename):
#     print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
#     try:
#         print("######################################")
#         uploads = os.path.join(current_app.root_path, "OntoData") # change with a app.config thing
#         print(uploads)

#         # f = open(uploads + '/' + filename + '1' , 'w')
#         # f.write(a)
#         # print(repr(a))
#         # f.close()
#         return send_from_directory(uploads, filename,as_attachment=True, attachment_filename=filename)
#     except Exception as e:
#         return str(e)

#     # @app.route('/uploadfile/')
#     # def upload_files(filename) :
#     #     try :
#     # @app.route('/uploadFile/', methods=['GET', 'POST'])
#     # def upload_file():
#     #     if request.method == 'POST':
#     #         # check if the post request has the file part
#     #         if 'file' not in request.files:
#     #             flash('No file part')
#     #             return redirect(request.url)
#     #         file = request.files['file']
#     #         # if user does not select file, browser also
#     #         # submit an empty part without filename
#     #         if file.filename == '':
#     #             flash('No selected file')
#     #             return redirect(request.url)
#     #         if file:
#     #             filename = secure_filename(file.filename)
#     #             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#     #             return redirect(url_for('uploaded_file',
#     #                                     filename=filename))
