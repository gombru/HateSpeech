import random
import json
import re

ids_file_path = '../../../datasets/HateSPic/AMT/50K/ids_50000_amt.csv'
out_file_path = '../../../datasets/HateSPic/AMT/MMHS2/MMHS50K_offline_amt_50k.csv'

tweets_data_path = '../../../datasets/HateSPic/MMHS/anns/MMHS50K_GT.json'
print("Loading tweets data")
tweets_data = json.load(open(tweets_data_path))


urls = open(ids_file_path).read().split("\n")
num_ids = []
for url in urls:
    if len(url) > 10:
        id = url.split('/')[-1]
        num_ids.append(int(id))

random.shuffle(num_ids)
out_file = open(out_file_path,'w')
out_file.write('tweet_url,img_url,tweet_text')


for i,id in enumerate(num_ids):
    # if i == 4800: break
    # i+=1

    tweet_url = "https://twitter.com/user/status/" + str(id)
    tweet_url = tweet_url.replace(',','').replace('\n','')
    img_url = "http://158.109.9.237:45993/data/MMHS/" + str(id) + '.jpg'
    tweet_text = tweets_data[str(id)]["tweet_text"].replace(',','').replace('\n','').replace('\r', '').encode('ascii','ignore').encode('utf-8','ignore')
    tweet_text = ''.join(char for char in tweet_text if len(char.encode('utf-8')) < 3)
    tweet_text = tweet_text.replace('"','')
    out_file.write('\n' + tweet_url + ',' + img_url + ',' + tweet_text)

out_file.close()