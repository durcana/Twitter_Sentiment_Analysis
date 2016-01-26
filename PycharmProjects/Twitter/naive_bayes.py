import nltk
import pickle
import random
from nltk.corpus import twitter_samples


class Memoize(object):
    def __init__(self, func):
        self.func = func
        self.memo = {}

    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.func(*args)
        return self.memo[args]

    def load_memo(self, filename):
        with open(filename, 'rb') as fp:
            self.memo.update(pickle.load(fp))

    def return_memo(self):
        return self.memo


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

    pickled = {'tweets': tweets,
               'word_features': word_features}

    with open('word_delegation.pickle', 'wb') as fp:
        pickle.dump(pickled, fp)


def delegation_mem():
    delegation_memoize = Memoize(word_delegation())
    delegation_memoize.load_memo('word_delegation.pickle')
    memo_dict = delegation_memoize.return_memo()
    return memo_dict['tweets'], memo_dict['word_features']


def find_features(tweet):
    tweets, word_features = delegation_mem()
    words = set(tweet)
    return {w: w in words for w in word_features}


def bayes(tweets, count):
    feature_sets = [(find_features(tweet), category) for (tweet, category) in tweets]
    random.shuffle(feature_sets)
    training_set = feature_sets[:count]
    testing_set = feature_sets[count:]

    classifier = nltk.NaiveBayesClassifier.train(training_set)

    print("Naive Bayes Algorithm accuracy percent:", nltk.classify.accuracy(classifier, testing_set)*100)
    # The pos : neg ratio that this comes out in would solve the average sentiment for my keywords.
    # Need to find how to do this for specific words, instead of the top however many words.
    print(classifier.show_most_informative_features(15))

    pickled = {'classifier': classifier}

    with open('bayes_classifier.pickle', 'wb') as fp:
        pickle.dump(pickled, fp)

    return classifier


def bayes_mem():
    tweets, word_features = delegation_mem()
    bayes_memoize = Memoize(bayes(tweets, count=1900))
    bayes_memoize.load_memo('bayes_classifier.pickle')
    memo_dict = bayes_memoize.return_memo()
    return memo_dict['classifier']


def sentiment(text):
    classifier = bayes_mem()
    features = find_features(text)
    return classifier.classify(features)

print(sentiment('I hope this works!'))
