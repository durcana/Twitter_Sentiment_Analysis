from json import loads
from tweepy import OAuthHandler
import os
import stopit
from tweepy import Stream
from tweepy.streaming import StreamListener
import naive_bayes_pickled


ckey = os.environ.get('CKEY')
csecret = os.environ.get('CSECRET')
atoken = os.environ.get('ATOKEN')
asecret = os.environ.get('ASECRET')

feed = open('twitter_feed.txt', 'a')
feed.write('[')
feed.close()


class Listener(StreamListener):

    def on_data(self, data):
        json_data = loads(data)
        tweet = json_data['text']

        if 'conviction' in tweet or 'birth' in tweet:
            sentiment = naive_bayes_pickled.sentiment(tweet)
            print tweet, sentiment
            output = open('twitter_feed.txt', 'a')
            output.write(tweet.encode('utf-8'))
            output.write('\n')
            output.close()
        return True

    def on_error(self, status):
        print status


auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
twitterStream = Stream(auth, Listener())
twitterStream.filter(track=['conviction', 'birth'], languages=['en'])

stopit.ThreadingTimeout(5)

feed = open('twitter_feed.txt', 'a')
feed.write(']')
feed.close()
