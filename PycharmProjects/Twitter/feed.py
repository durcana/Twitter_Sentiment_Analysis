from json import loads
from tweepy import OAuthHandler
import os
from tweepy import Stream
from tweepy.streaming import StreamListener


ckey = os.environ.get('CKEY')
csecret = os.environ.get('CSECRET')
atoken = os.environ.get('ATOKEN')
asecret = os.environ.get('ASECRET')


class Listener(StreamListener):

    def on_data(self, data):
        json_data = loads(data)
        tweet = json_data['text']

        if 'conviction' in tweet or 'birth' in tweet:
            print tweet
            output = open('twitterfeed.txt', 'a')
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
