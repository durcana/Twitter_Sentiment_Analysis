import nltk
import pickle
import random
from nltk.corpus import twitter_samples


# Still figuring out how to do the memoization. Labled 2 functions as memoize 1 and 2 to define where it needs work.
# I did bring it into one file though and got the script to flow much better.
# Once the memoize fuctions are fixed, feed.py should run sentiment no problem.

info = {}


def memoize1():
    if w_feats in info:
        with open('word_delegation.pickle', 'rb') as fp:
            pickled = pickle.load(fp)
            tweets = pickled[0]
            word_features = pickled[1]
        return tweets, word_features
    else:
        tweets, word_features = word_delegation()
        info['w_feats'] = 'pickled'
        return tweets, word_features


def word_delegation():
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

    pickled = [tweets, word_features]

    with open('word_delegation.pickle', 'wb') as fp:
        pickle.dump(pickled, fp)

    return tweets, word_features


def find_features(tweet):
    tweets, word_features = memoize1()
    words = set(tweet)
    return {w: w in words for w in word_features}


def memoize2():
    if classifier in info:
        with open('bayes_classifier.pickle', 'rb') as fp:
            classifier = pickle.load(fp)
        return classifier
    else:
        bayes(tweets, count=1900)
        info['classifier'] = 'pickled'


def bayes(tweets, count):
    feature_sets = [(find_features(tweet), category) for (tweet, category) in tweets]
    random.shuffle(feature_sets)
    training_set = feature_sets[:count]
    testing_set = feature_sets[count:]

    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print ("Naive Bayes Algorithm accuracy percent:", nltk.classify.accuracy(classifier, testing_set)*100)

    # The pos : neg ratio that this comes out in would solve the average sentiment for my keywords.
    # Need to find how to do this for specific words, instead of the time however many words.
    print (classifier.show_most_informative_features(15))

    with open('bayes_classifier.pickle', 'wb') as fp:
        pickle.dump(classifier, fp)

    return classifier


def sentiment(text):
    classifier = memoize2()
    features = find_features(text)
    return classifier.classify(features)
