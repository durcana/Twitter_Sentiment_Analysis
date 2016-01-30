import nltk
import random
from nltk.corpus import twitter_samples


DELEGATION_RESULTS = None


def word_delegation():
    positive = twitter_samples.tokenized('positive_tweets.json')
    negative = twitter_samples.tokenized('negative_tweets.json')

    tweets, all_words = [], []

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

    return tweets, word_features


def find_features(tweet):
    global DELEGATION_RESULTS

    if DELEGATION_RESULTS is None:
        DELEGATION_RESULTS = word_delegation()

    tweets, word_features = DELEGATION_RESULTS

    words = set(tweet)

    return {w: w in words for w in word_features}


def bayes(count):
    tweets, word_features = word_delegation()
    feature_sets = [(find_features(tweet), category) for (tweet, category) in tweets]
    random.shuffle(feature_sets)
    training_set = feature_sets[:count]
    testing_set = feature_sets[count:]

    classifier = nltk.NaiveBayesClassifier.train(training_set)

    print("Naive Bayes Algorithm accuracy percent:", nltk.classify.accuracy(classifier, testing_set)*100)
    classifier.show_most_informative_features(15)

    return classifier


def sentiment(text):
    classifier = bayes(count=1900)
    features = find_features(text)
    return classifier.classify(features)
