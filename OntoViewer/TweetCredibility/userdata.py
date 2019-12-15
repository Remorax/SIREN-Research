from tweepy import OAuthHandler
import numpy as np
from tweepy import API
from tweepy import Cursor
from datetime import datetime, date, time, timedelta
from collections import Counter
import sys

access_token = "1192925360851013632-a1OH6gVyKWcmvMzeGkeQWNJYGGmQN9"
access_token_secret = "Kranny95fVLF5bn9pCrB4B2TXjM4oTnT9BX3vztxEPrDf"
consumer_key = "9NDG7eIVsrouj4CS2M7LoNjM1"
consumer_secret = "y1z075l563BwcL8XtI7GzQzEnvo1jEEzmcmR1NFBxhYPFokYzu"

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
auth_api = API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, retry_count=3, retry_delay=60)

account_list = []
if (len(sys.argv) > 1):
	account_list = sys.argv[1:]
else:
	print("Provide a list of usernames.")
	sys.exit(0)

def limit_handled(cursor):
	count=0
	while True:
		try:
			count+=1
			yield cursor.next()
		except tweepy.RateLimitError:
			print(count/2302*100)
			time.sleep(15 * 60)


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
		# 	frndlist.append(user.screen_name)

		print(len(frndlist[0]))
		lst.append((item.id, '\n'.join(tweets),'\n'.join(pop_tweets)))
	print("The Tweets extracted from target user:- \n\n")
	print(lst[0][1])
	print("\n\n\n\nThe top most popular tweets extracted from the following list accoutns. \n\n\n\n\n")
	print(lst[0][2])