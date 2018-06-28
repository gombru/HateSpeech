import twitter
import urllib
import time
from twython import Twython

CONSUMER_KEY = "uHmr7pmSU6yBiEtbpZQPSsqlQ"
CONSUMER_SECRET = "xICgXtFxp6HrQDQh2oAd6OysFxDoO9mo5blarLeBB8aegALrkH"
OAUTH_TOKEN = "81841533-PU84e9z6jNt1AtgHP13GnS8tRJGTMSJ3lLvMevYpE"
OAUTH_TOKEN_SECRET = "33ySOzqucOiMCst5dZcRcbyzKPjKx7xSNp9aj7esdCFa5"

t = twitter.Api(
    consumer_key=CONSUMER_KEY,
    consumer_secret=CONSUMER_SECRET,
    access_token_key=OAUTH_TOKEN,
    access_token_secret=OAUTH_TOKEN_SECRET,
    sleep_on_rate_limit=True
)

def get_replies(tweet_id, user):
    out_replies = []
    max_id = None
    while True:
        q = urllib.urlencode({"q": "to:%s" % user})
        try:
            replies = t.GetSearch(raw_query=q, since_id=tweet_id, max_id=max_id, count=100)
        except twitter.error.TwitterError as e:
            print("caught twitter api error: %s", e)
            time.sleep(60)
            continue
        for reply in replies:
            if reply.in_reply_to_status_id == tweet_id:
                out_replies.append(reply)
                # recursive magic to also get the replies to this reply
                # for reply_to_reply in get_replies(reply):
                #     yield reply_to_reply
            #max_id = reply.id
        if len(replies) != 100:
            break

    return replies


# They say that twitter only provides respones from last 7 days. SO I can get responses from the tweets I get, but maybe not from preannotated databases
# In case I use the tweets I capture, I can directly load from the json the id and the user. I can get responses to the ones classified as hate by the LSTM

tweet_id = 1012292365367873537
# Get tweet user
twython = Twython(CONSUMER_KEY, CONSUMER_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
tweet = twython.show_status(id=tweet_id)
user = tweet['user']['screen_name']
replies = get_replies(tweet_id, user)

print("Number of replies: " + str(len(replies)))
for r in replies: print r.text
print replies

print('Done')