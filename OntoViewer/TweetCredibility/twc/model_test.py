import pickle
import nltk

classifier = pickle.load(open('data/trained/MNB.pickle', 'rb'))
word_features = pickle.load(open('data/trained/word_features.pickle', 'rb'))

def document_features(document):
	document_words = set(document)
	features = {}
	for word in word_features:
		features['contains(%s)' % word] = (word in document_words)
	return features

def predict_topic(s):
	token = nltk.word_tokenize(s.lower())
	return classifier.classify(document_features(token))



topic = predict_topic("""Bots are software programs created to automatically perform specific operations. While some bots are created for relatively harmless purposes (video gaming, internet auctions, online contests, etc), it is becoming increasingly common to see bots being used maliciously.""")

print(topic)