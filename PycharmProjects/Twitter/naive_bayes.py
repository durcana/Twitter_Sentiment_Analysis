import nltk
import pickle
import random
from nltk.corpus import twitter_samples


def main():
    tweets, word_features = word_delegation()
    testing_set, training_set = bayes_parameters(tweets, word_features)
    classifier = bayes(testing_set, training_set)
    pickling(word_features, classifier)


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


def bayes(testing_set, training_set):
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print ("Naive Bayes Algorithm accuracy percent:", nltk.classify.accuracy(classifier, testing_set)*100)
    print (classifier.show_most_informative_features(15))

    return classifier


def pickling(word_features, classifier):
    save_word_features = open('bayes_word_features.pickle', 'wb')
    pickle.dump(word_features, save_word_features)
    save_word_features.close()

    save_classifier = open('bayes_classifier.pickle', 'wb')
    pickle.dump(classifier, save_classifier)
    save_word_features.close()


if __name__ == '__main__':
    main()
