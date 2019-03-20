from preprocess_tweets import tweet_preprocessing
import random
import os
import json

base_path = '../../../../datasets/HateSPic/twitter/json_all/'
out_path = '../../../../datasets/HateSPic/lstm_data/twitter/'

out_file = open(out_path + 'tweets.test', 'w')

for file in os.listdir(base_path):
    print file
    info = json.load(open(base_path + file))
    try:
        text = tweet_preprocessing(info['text'].encode('utf-8'))
        # Discard short tweets
        if len(text) < 5: continue
        if len(text.split(' ')) < 3: continue
        text = text.strip('\r').strip('\n')
        out_file.write(str(info['id']) + ',' + text +'\n')

    except:
        print("Error with file: " + file)
        continue

print "DONE"