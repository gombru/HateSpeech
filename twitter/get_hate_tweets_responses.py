import twitter
import urllib
import time
from twython import Twython, TwythonRateLimitError, TwythonError

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
file_path = '../../../datasets/HateSPic/twitter/lstm_scores_1_2.txt'
out_file_path = open('../../../datasets/HateSPic/twitter/hate_responses_1_2.txt','w')
hate_threshold = 0.7
total_replies = 0
tweet_ids = []
for line in open(file_path, 'r'):
    if float(line.split(',')[1]) > hate_threshold:
        tweet_id = int(line.split(',')[0])
        tweet_ids.append(tweet_id)

print("Getting responses for " + str(len(tweet_ids))) + " tweets"
# tweet_id = 1012292365367873537
# Get tweet user
twython = Twython(CONSUMER_KEY, CONSUMER_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)

for cur_tweet_id in tweet_ids:
    try:
        tweet = twython.show_status(id=cur_tweet_id)
        user = tweet['user']['screen_name']
        replies = get_replies(cur_tweet_id, user)
        print("Number of replies: " + str(len(replies)))
        total_replies += len(replies)
        for r in replies:
            out_file_path.write(str(cur_tweet_id) + ',' + str(r.id) + ',' + r.text.encode("utf8", "ignore").replace('\n', ' ').replace('\r', ' ') + '\n')
        #     print r.text
        # print replies
    except TwythonRateLimitError as error:
        remainder = float(twython.get_lastfunction_header(header='x-rate-limit-reset')) - time.time()
        print("Rate limit reched, sleeping (s): " + str(remainder))
        del twython
        if remainder <= 0: remainder = 1
        time.sleep(remainder)
        twython = Twython(CONSUMER_KEY, CONSUMER_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
        print("Resuming -->")
        continue
    except TwythonError as error:
        print "Tweet does not exist: " + str(cur_tweet_id)
        continue

out_file_path.close()
print("Toral number of replies: " + str(total_replies))
print('Done')