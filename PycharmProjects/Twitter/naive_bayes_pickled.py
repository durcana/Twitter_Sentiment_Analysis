import nltk
import pickle
import random
from nltk.corpus import twitter_samples


def main():
    tweets, word_features = word_delegation()
    testing_set = bayes_parameters(tweets, word_features)
    bayes(testing_set)


def word_delegation():
    positive = twitter_samples.tokenized('positive_tweets.json')
    negative = twitter_samples.tokenized('negative_tweets.json')

    tweets = []
    for tweet in positive:
        tweets.append((tweet, 'pos'))
    for tweet in negative:
        tweets.append((tweet, 'neg'))

    saved_word_features = open('bayes_word_features.pickle', 'rb')
    word_features = pickle.load(saved_word_features)
    saved_word_features.close()

    return tweets, word_features


def bayes_parameters(tweets, word_features):

    # I feel like there has to be a better way to do this to suffice feature_sets definition
    def find_features():
        words = set(tweets)
        features = {}
        for w in word_features:
            features[w] = (w in words)

        return features

    feature_sets = [(find_features(), category) for (tweet, category) in tweets]
    random.shuffle(feature_sets)
    training_set = feature_sets[:1900]
    testing_set = feature_sets[1900:]

    return testing_set, training_set


def bayes(testing_set):
    saved_classifier = open('bayes_classifier.pickle', 'rb')
    classifier = pickle.load(saved_classifier)
    saved_classifier.close()

    print ("Naive Bayes Algorithm accuracy percent:", nltk.classify.accuracy(classifier, testing_set)*100)
    print (classifier.show_most_informative_features(15))

    return classifier


def find_features(tweet_list):
    _tweets, word_features = word_delegation()
    words = set(tweet_list)
    return {w: w in words for w in word_features}


FEATURES = find_features()


def sentiment(text):
    """
    Return sentiment of tweet
    """
    return classifier.classify(FEATURES)


# Do I need this every time?
if __name__ == '__main__':
    main()


# When testing the streamed data, I figured I'd want the same data that classifier.show_most_informative_features gives.
# It shows in format:         sad = True    neg : pos = 12.6 : 1.0
# I could then use the neg:pos ratio to calculate the average sentiment.

# What I'm trying to now figure out is out to incorporate the streamed data from the txt file into this algorithm.
