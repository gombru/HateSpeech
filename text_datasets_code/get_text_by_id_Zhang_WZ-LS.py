from twython import Twython, TwythonRateLimitError, TwythonError
import os
from PIL import Image
import urllib
import time

CONSUMER_KEY = "uHmr7pmSU6yBiEtbpZQPSsqlQ"
CONSUMER_SECRET = "xICgXtFxp6HrQDQh2oAd6OysFxDoO9mo5blarLeBB8aegALrkH"
OAUTH_TOKEN = "81841533-PU84e9z6jNt1AtgHP13GnS8tRJGTMSJ3lLvMevYpE"
OAUTH_TOKEN_SECRET = "33ySOzqucOiMCst5dZcRcbyzKPjKx7xSNp9aj7esdCFa5"
twitter = Twython(CONSUMER_KEY, CONSUMER_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)

ids = []
classes = []

file = open('../../../datasets/HateSPic/Zhang/wz-ls/labeled_data.csv', 'r')
out_file = open('../../../datasets/HateSPic/Zhang/wz-ls/labeled_data_text.csv','w')


for line in file:
    ids.append(int(line.split(',')[0]))
    t_class = 0
    if int(line.split(',')[1]) == 2:
        t_class = 2
    classes.append(t_class)

corrupted = 0
saved_txts = 0

for i,id in enumerate(ids):

    print(str(i) + " Downloaded txt: " + str(saved_txts))

    try:
        t = twitter.show_status(id=id)

    except TwythonRateLimitError as error:
        remainder = float(twitter.get_lastfunction_header(header='x-rate-limit-reset')) - time.time()
        print("Rate limit reched, sleeping (s): " + str(remainder))
        del twitter
        if remainder <= 0: remainder = 1
        time.sleep(remainder)
        twitter = Twython(CONSUMER_KEY, CONSUMER_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
        print("Resuming -->")
        continue

    except TwythonError as error:
        print "Tweet does not exist: " + str(id)
        corrupted += 1
        continue

    try:
        out_file.write(str(classes[i]) + ',' + t['text'] + '\n')
        saved_txts += 1
    except:
        print("Failed writing text")
        continue

    print "done"

out_file.close()
print("Corrupted: " + str(corrupted))
print("Saved txt: " + str(saved_txts))
