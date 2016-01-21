import nltk
import pickle
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

saved_word_features = open('bayes_word_features.pickle', 'rb')
word_features = pickle.load(saved_word_features)
saved_word_features.close()


def find_features(tweets_list):
    words = set(tweets_list)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


feature_sets = [(find_features(tweet), category) for (tweet, category) in tweets]
random.shuffle(feature_sets)
training_set = feature_sets[:1900]
testing_set = feature_sets[1900:]

saved_classifier = open('bayes_classifier.pickle', 'rb')
classifier = pickle.load(saved_classifier)
saved_word_features.close()

print "Naive Bayes Algorithm accuracy percent:", nltk.classify.accuracy(classifier, testing_set)*100
classifier.show_most_informative_features(15)


def sentiment(text):
    feats = find_features(text)

    return classifier.classify(feats)


print sentiment("birth")
print sentiment("conviction")


# When testing the streamed data, I figured I'd want the same data that classifier.show_most_informative_features gives.
# It shows in format:         sad = True    neg : pos = 12.6 : 1.0
# Then use the neg:pos ratio to calculate the average sentiment.

# What I'm trying to now figure out is out to incorporate the streamed data from the txt file into this algorithm.

# Also, SentimentAnalyzer has modules which can be seen at:
# http://www.nltk.org/_modules/nltk/sentiment/sentiment_analyzer.html#SentimentAnalyzer
# Should I use SentimentAnalyzer to make any of this simpler?
