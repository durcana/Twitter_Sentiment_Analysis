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

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:3000]

save_word_features = open('bayes_word_features.pickle', 'wb')
pickle.dump(word_features, save_word_features)
save_word_features.close()


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

classifier = nltk.NaiveBayesClassifier.train(training_set)
print "Naive Bayes Algorithm accuracy percent:", nltk.classify.accuracy(classifier, testing_set)*100
classifier.show_most_informative_features(15)

save_classifier = open('bayes_classifier.pickle', 'wb')
pickle.dump(classifier, save_classifier)
save_word_features.close()


def sentiment(text):
    feats = find_features(text)

    return classifier.classify(feats)
