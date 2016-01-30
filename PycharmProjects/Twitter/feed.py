from json import loads
from tweepy import OAuthHandler
import os
from tweepy import Stream
from tweepy.streaming import StreamListener
import naive_bayes


ckey = os.environ.get('CKEY')
csecret = os.environ.get('CSECRET')
atoken = os.environ.get('ATOKEN')
asecret = os.environ.get('ASECRET')


class Listener(StreamListener):

    def on_data(self, data):
        json_data = loads(data)
        tweet = json_data['text']

        if 'conviction' in tweet or 'birth' in tweet:
            print(tweet)
            print(naive_bayes.sentiment(tweet))
        return True

    def on_error(self, status):
        print(status)


def main():
    auth = OAuthHandler(ckey, csecret)
    auth.set_access_token(atoken, asecret)
    twitter_stream = Stream(auth, Listener())
    twitter_stream.filter(track=['conviction', 'birth'], languages=['en'])


if __name__ == '__main__':
    main()
