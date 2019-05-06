import random
import json
import re
import os

jsons_path = '../../../datasets/HateSPic/AMT/MMHS2/2label_extra/'
out_file = open('../../../datasets/HateSPic/AMT/MMHS2/MMHS100K_offline.csv','w')

print("Loading tweets data")
data = {}
for file in os.listdir(jsons_path):
    id = int(file.split('/')[-1].split('.')[0])
    d = json.load(open(jsons_path + file))
    data[id] = d['text']


for k,v in data.iteritems():
    # if i == 4800: break
    # i+=1
    tweet_url = "https://twitter.com/user/status/" + str(k)
    tweet_url = tweet_url.replace(',','').replace('\n','')
    img_url = "http://158.109.9.237:45993/data/MMHS/" + str(k) + '.jpg'
    tweet_text = v.replace(',','').replace('\n','').replace('\r', '').encode('ascii','ignore').encode('utf-8','ignore')
    tweet_text = ''.join(char for char in tweet_text if len(char.encode('utf-8')) < 3)
    tweet_text = tweet_text.replace('"','')
    out_file.write('\n' + tweet_url + ',' + img_url + ',' + tweet_text)

out_file.close()