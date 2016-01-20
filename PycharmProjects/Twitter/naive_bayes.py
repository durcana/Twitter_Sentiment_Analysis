import nltk
import random
from nltk.corpus import twitter_samples


positive = twitter_samples.tokenized('positive_tweets.json')
negative = twitter_samples.tokenized('negative_tweets.json')

tweets = []
all_words = []
for tweet in positive:
    tweets.append((tweet, 'pos'))
    for word in tweet:
        all_words.append(word.lower())
for tweet in negative:
    tweets.append((tweet, 'neg'))
    for word in tweet:
        all_words.append(word.lower())

random.shuffle(tweets)

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:3000]


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        # reads as word = True/False
        features[w] = (w in words)

    return features


feature_sets = [(find_features(tweet), category) for (tweet, category) in tweets]
training_set = feature_sets[:1900]
testing_set = feature_sets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print "Naive Bayes Algorithm accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100
classifier.show_most_informative_features(15)

# When testing the streamed data, I figured I'd want the same data that classifier.show_most_informative_features gives.
# It shows as:           sad = True    neg : pos = 12.6 : 1.0
# Then use the neg:pos ratio to calculate the average sentiment.

# What I'm trying to now figure out is out to incorporate the streamed data from the txt file into this algorithm.

# Also, SentimentAnalyzer has modules which can be seen at:
# http://www.nltk.org/_modules/nltk/sentiment/sentiment_analyzer.html#SentimentAnalyzer
# Should I use SentimentAnalyzer to make any of this simpler?
