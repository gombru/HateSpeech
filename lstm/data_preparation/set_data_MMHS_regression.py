from preprocess_tweets import tweet_preprocessing
import json
import random


data_path = '../../../../datasets/HateSPic/MMHS/anns/50k_3workers.json'
out_path = '../../../../datasets/HateSPic/MMHS/lstm_data/lstm_data_50k_3workers_regression/'

out_file_train = open(out_path + 'tweets.train', 'w')
out_file_val = open(out_path + 'tweets.val', 'w')
out_file_test = open(out_path + 'tweets.test', 'w')

print("Loading data ...")
data = json.load(open(data_path,'r'))

train = []
val = []
test = []

print("Generating lstm data")
for k,v in data.iteritems():
    total_hate = 0
    for label in v['labels']:
        if label > 0:
            total_hate += 1

    label = total_hate / 3.0

    text = tweet_preprocessing(v['tweet_text'].encode('utf-8'))

    split_selector = random.randint(1,10)

    if split_selector > 8:
        val.append(str(k) + ',' + text + ',' + str(label) + '\n')
    elif split_selector > 7:
        test.append(str(k) + ',' + text + ',' + str(label) + '\n')
    else:
        train.append(str(k) + ',' + text + ',' + str(label) + '\n')

for l in train: out_file_train.write(l)
for l in val: out_file_val.write(l)
for l in test: out_file_test.write(l)

print("DONE")