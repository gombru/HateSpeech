# encoding=utf8
import os
import json
from PIL import Image
import urllib
import random
import sys

base_path = '../../../datasets/HateSPic/AMT/50K/2label2/'

out_file = open('../../../datasets/HateSPic/AMT/50K/ids_50000_amt.csv','w')
out_file.write('tweet_url\n')

existing_file = open('../../../datasets/HateSPic/AMT/50K/ids_48000_amt.csv','r')
used_ids = []
for i,line in enumerate(existing_file):
    if i == 0: continue
    id = int(line.split('/')[-1].strip('\n'))
    used_ids.append(id)
print("Used ids: " + str(len(used_ids)))


def download_save_image(url, image_path):
    resource = urllib.urlopen(url)
    output = open(image_path, "wb")
    output.write(resource.read())
    output.close()

tweets = []
idx=0
nigga_tweets = 0
written_tweets = 0
max_nigga_tweets = 500
tweets_2_write = 50000 - len(used_ids)
print("Checking if tweets still exists ...")
for file in os.listdir(base_path):
    if written_tweets == tweets_2_write: break
    i = json.load(open(base_path + file, 'r'))
    idx+=1

    if int(i['id']) in used_ids:
        print("Continuing")
        continue

    if 'nigga' in i['text'] or 'Nigga' in i['text']:
        if nigga_tweets < max_nigga_tweets:
            nigga_tweets += 1
        else:
            continue

    try:
        download_save_image(i['img_url'], "/home/raulgomez/datasets/img_test.jpg")
        im = Image.open("/home/raulgomez/datasets/img_test.jpg")
        tweets.append(i)
        written_tweets+=1
        print(written_tweets)
    except:
        print("Error")
        continue

print "Total tweets: " + str(len(tweets))

random.shuffle(tweets)

print("Writing")
for id in used_ids:
    out_file.write("https://twitter.com/user/status/" + str(id) + '\n')
for t in tweets:
    out_file.write("https://twitter.com/user/status/" + str(t['id']) + '\n')






