# encoding=utf8
import os
import json
from PIL import Image
import urllib
import random
import sys

out_file = open('../../../datasets/HateSPic/AMT/ids_50000_amt.csv','w')
out_file.write('tweet_id\n')

base_path = '../../../datasets/HateSPic/AMT/2label/'


def download_save_image(url, image_path):
    resource = urllib.urlopen(url)
    output = open(image_path, "wb")
    output.write(resource.read())
    output.close()

tweets = []
idx=0
nigga_tweets = 0
written_tweets = 0
print("Checking if tweets still exists ...")
for file in os.listdir(base_path):
    if written_tweets == 50000: break
    i = json.load(open(base_path  + file, 'r'))
    idx+=1

    if 'nigga' in i['text'] or 'Nigga' in i['text']:
        if nigga_tweets < 4000:
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
for t in tweets:
    out_file.write("https://twitter.com/user/status/" + str(t['id']) + '\n')






