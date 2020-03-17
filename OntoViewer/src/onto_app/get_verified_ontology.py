import sys
import subprocess
from os import listdir
from os.path import isfile, join
import xml.dom.minidom
from owlready2 import *
from aggregate import generate_final_ontology,delete_nodes
from gensim.models import KeyedVectors
import tweepy
from tweepy import OAuthHandler
import numpy as np
from tweepy import API
from tweepy import Cursor
from datetime import datetime, date, time, timedelta
from collections import Counter 
import string
model = KeyedVectors.load_word2vec_format("~/word2vec_twitter_model.bin", binary=True, unicode_errors='ignore')
import sys
from collections import defaultdict
from sqlalchemy import create_engine
name = sys.argv[1]
print (name)

def generateScore(text):
    unpunctuatedText = text.translate(str.maketrans('', '', string.punctuation))
    words = unpunctuatedText.split(" ")
    totalScore = 0
    count = 0
    for word in words:
        try:
            sim1 = model.similarity(word, "pizza")
            sim2 = model.similarity(word, "garlicbread")
            sim3 = model.similarity(word, "toppings")
            # # # sim4 = model.similarity(word, "cryptography")
            score = (sim1 + sim2 + sim3)/3
            totalScore += score
            count += 1
            # totalScore = 200
            # count = 300
        except:
            continue
    try:     
        totalScore = totalScore/count
        return totalScore
    except:
        return 0
    


access_token = "1192925360851013632-a1OH6gVyKWcmvMzeGkeQWNJYGGmQN9"
access_token_secret = "Kranny95fVLF5bn9pCrB4B2TXjM4oTnT9BX3vztxEPrDf"
consumer_key = "9NDG7eIVsrouj4CS2M7LoNjM1"
consumer_secret = "y1z075l563BwcL8XtI7GzQzEnvo1jEEzmcmR1NFBxhYPFokYzu"

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
auth_api = API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, retry_count=3, retry_delay=60)

account_list = []
# if (len(sys.argv) > 1):
#     account_list = sys.argv[1:]
# else:
#     print("Provide a list of usernames.")
#     sys.exit(0)
def add_relation_with_credibility_only(twitter_users):
    # query = """SELECT * FROM class_decisions"""
    # result = db.engine.execute(query)
    engine = create_engine('sqlite:///onto.db', echo = True)
    # conn = sqlite3.connect('onto.db')
    # c = conn.cursor()
    c = engine.connect()
    trans = c.begin()
    print("XYZ_1")
    query = """SELECT * FROM class_decisions INNER JOIN class_relations ON class_decisions.relation_id =class_relations.id """
    result = c.execute(query)
    relation_list = [(o['relation_id'],o['property'],o['domain'],o['range']) for o in result.fetchall()]
    # relation_set = set(relation_list)
    relation_dict = defaultdict(int)
    relation_count = defaultdict(int)
    query = """SELECT * FROM class_decisions INNER JOIN class_relations ON class_decisions.relation_id =class_relations.id """
    result = c.execute(query)
    print(relation_list)
    # print(result.fetchall())
    result = list(result.fetchall())
    print(result)
    for tup in relation_list:
        print(tup)
        for o in result:
            print("TATA")
            print(o['property'])
            print(o['domain'])
            print(o['range'])
            print(tup[1])
            print(tup[2])
            print(tup[3])
            if(tup[1] == o['property'] and tup[2] == o['domain'] and tup[3] == o['range']):
                print("Yo")
                relation_dict[tup]+=(o['approved']*twitter_users[o['user_id']])
                print(twitter_users[o['user_id']])
                relation_count[tup]+=twitter_users[o['user_id']]
            else:
                pass


    query = """DELETE  FROM class_decisions """
    result = c.execute(query)
    print("XTZ")
    print(relation_dict)
    for tup,score in relation_dict.items():
        print("PPP")
        score = score/relation_count[tup]
        relation_dict[tup] = score
    for tup,score in relation_dict.items():
        print("PPP")
            # print(tup)
        if score > 0.5:
            print(tup)
            args = {
                        'relation_id': tup[0],
                            # 'property': property,
                        'approved': 1,
                        'user_id': 12345678
                        }
            print(args)
            insert_query = """INSERT INTO class_decisions
                            (relation_id, user_id, approved)
                            VALUES (:relation_id, :user_id, :approved)"""
            c.execute(insert_query,args)
        else:
            print(tup)
            args = {
                        'relation_id': tup[0],
                            # 'property': property,
                        'approved': 0,
                        'user_id': 12345678
                        }
            insert_query = """INSERT INTO class_decisions
                            (relation_id, user_id, approved)
                            VALUES (:relation_id, :user_id, :approved)"""
            c.execute(insert_query,args)
    print("XYZ_1")
    query = """SELECT * FROM node_decisions INNER JOIN nodes ON node_decisions.node_id =nodes.id """
    result = c.execute(query)
    nodes_list = [(o['node_id'],o['name']) for o in result.fetchall()]
    # relation_set = set(relation_list)
    relation_dict = defaultdict(int)
    relation_count = defaultdict(int)
    query = """SELECT * FROM node_decisions INNER JOIN nodes ON node_decisions.node_id =nodes.id """
    result = c.execute(query)
    print(relation_list)
    # print(result.fetchall())
    result = list(result.fetchall())
    print(result)
    for tup in nodes_list:
        print(tup)
        for o in result:
            print("TATA")
            print(o['node_id'])
            print(o['name'])
            print(tup[0])
            print(tup[1])
            if(tup[0] == o['node_id'] and tup[1] == o['name']):
                relation_dict[tup]+=(o['approved']*twitter_users[o['user_id']])
                relation_count[tup]+=twitter_users[o['user_id']]
            else:
                pass


    query = """DELETE  FROM node_decisions """
    result = c.execute(query)
    print("XTZ")
    print(relation_dict)
    for tup,score in relation_dict.items():
        print("PPP")
        score = score/relation_count[tup]
        relation_dict[tup] = score
    for tup,score in relation_dict.items():
        print("PPP")
            # print(tup)
        if score > 0.5:
            print(tup)
            args = {
                        'node_id': tup[0],
                            # 'property': property,
                        'approved': 1,
                        'user_id': 12345678
                        }
            print(args)
            insert_query = """INSERT INTO node_decisions
                            (node_id, user_id, approved)
                            VALUES (:node_id, :user_id, :approved)"""
            c.execute(insert_query,args)
        else:
            print(tup)
            args = {
                        'node_id': tup[0],
                            # 'property': property,
                        'approved': 0,
                        'user_id': 12345678
                        }
            insert_query = """INSERT INTO node_decisions
                            (node_id, user_id, approved)
                            VALUES (:node_id, :user_id, :approved)"""
            c.execute(insert_query,args)
            
    trans.commit()

def limit_handled(cursor):
    count=0
    while True:
        try:
            count+=1
            yield cursor.next()
        except tweepy.RateLimitError:
            print(count/2302*100)
            time.sleep(15 * 60)

engine = create_engine('sqlite:///onto.db', echo = True)
    # conn = sqlite3.connect('onto.db')
    # c = conn.cursor()
c = engine.connect()
trans = c.begin()
query = """SELECT * FROM users"""
result = c.engine.execute(query)
for o in result.fetchall():
    account_list.append(o['username'])

account_list = list(set(account_list))    
if len(account_list) > 0:
    lst = []
    for target in account_list:
        print("Getting data for " + target)
        item = auth_api.get_user(target)
        print("name: " + item.name)
        print("screen_name: " + item.screen_name)
        print("description: " + item.description)
        print("statuses_count: " + str(item.statuses_count))
        print("friends_count: " + str(item.friends_count))
        print("followers_count: " + str(item.followers_count))
        tweets_data = auth_api.user_timeline(screen_name=item.screen_name, count=200, tweet_mode="extended")
        tweets = [tweet.full_text for tweet in tweets_data]
        frndlist = []
        # user = auth_api.get_user(item.screen_name)
        # for usr in user.friends():
            # frndlist.append(usr)
        req_dict={}
        pop_tweets = [] 
        for follower in Cursor(auth_api.friends_ids,screen_name=item.screen_name).pages():
            # if (follower.friends_count<300):
            frndlist.append(follower)
            # print(np.array(follower).shape)
            # print(follower)
            # for num,els in enumeratefollower[0]:
            for i in range(10):
                friend_user=auth_api.get_user(user_id=follower[i])
                for tweet in Cursor(auth_api.search, q='from:@'+friend_user.screen_name,result_type='popular').items(15):
                    pop_tweets.append(tweet.text)

                req_dict[friend_user.id] = pop_tweets
            # print(friend_user.screen_name)
        # req_dict[item.id]=
        # for user in Cursor(auth_api.friends, screen_name=item.screen_name).items():
        #   frndlist.append(user.screen_name)

        # print(len(frndlist[0]))
        lst.append((item.id, '\n'.join(tweets),'\n'.join(pop_tweets)))
    

finalDict = defaultdict(int)
for (userid, tweets, followers) in lst:
    finalDict[userid] = generateScore(tweets) + 10 * generateScore(followers)
print(finalDict)
add_relation_with_credibility_only(finalDict)


generate_final_ontology(name)

delete_nodes(name)