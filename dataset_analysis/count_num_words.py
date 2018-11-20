import os
import json
import operator

dir = "../../../datasets/HateSPic/AMT/2label/"
words = {}

c=0
for filename in os.listdir(dir):
    c+=1
    with open(dir + filename) as file:
        data = json.load(file)
        text = data['text']
        tweet_words = text.split(' ')
        for w in tweet_words:
            w = w.lower()
            if w in words:
                words[w] += 1
            else:
                words[w] = 1
    # if c == 10000: break

sorted_d = sorted(words.items(), key=operator.itemgetter(1))
for i in range(1,100):
    print sorted_d[-i]

print "Total tweets: " + str(c)

