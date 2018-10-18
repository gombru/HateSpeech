# encoding=utf8
import os
import json
from PIL import Image
import urllib
import random
import sys

out_file = open('../../../datasets/HateSPic/HateSPic/AMT/100_amt_test2.csv','w')
out_file2 = open('../../../datasets/HateSPic/HateSPic/AMT/100_amt_test2_text.csv','w')

base_path = '../../../datasets/HateSPic/HateSPicLabeler/generated_json_v2mm/'
datasets = ['HateSPic/','SemiSupervised/','WZ-LS/']

hate_count = 0
mm_hate_count = 0
nothate_count = 0
mm_hate_tweets = []


def download_save_image(url, image_path):
    resource = urllib.urlopen(url)
    output = open(image_path, "wb")
    output.write(resource.read())
    output.close()

for dataset in datasets:
    for file in os.listdir(base_path + dataset):
        i = json.load(open(base_path + dataset + file, 'r'))

        if 'mm_hate_votes' in i:
            if i['mm_hate_votes'] > 0:
                try:
                    download_save_image(i['img_url'], "/home/rgomez/datasets/img_test.jpg")
                    im = Image.open("/home/rgomez/datasets/img_test.jpg")
                    mm_hate_count += 1
                    mm_hate_tweets.append(i)
                except:
                    continue
            elif i['hate_votes'] > i['not_hate_votes'] and i['hate_votes'] > i['mm_hate_votes']:
                hate_count += 1
            else:
                nothate_count += 1
        else:
            nothate_count += 1


print "Total hate: " + str(hate_count) + " Total mm hate: " + str(mm_hate_count) + " Total NotHate: " + str(nothate_count)

random.shuffle(mm_hate_tweets)

reload(sys)
sys.setdefaultencoding('utf8')

out_file.write('tweet_id\n')
out_file2.write('content,tweet_id\n')
for i,e in enumerate(mm_hate_tweets):
    out_file.write("https://twitter.com/user/status/" + str(e['id']) + '\n')
    text = e['text'].encode('utf-8')
    words = text.split(' ')
    text = ''
    for w in words:
        if 'http' in w: continue
        text+= w + ' '
    text = text.replace(',','')
    text = text.replace('"','')
    text = text.encode('ascii', 'ignore')
    out_file2.write(text + ',' + str(e['id']) + '\n')
    if i == 99: break





