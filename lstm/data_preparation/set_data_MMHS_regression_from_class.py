from preprocess_tweets import tweet_preprocessing
import json
import random


data_path = '../../../../datasets/HateSPic/MMHS/anns/50k_3workers.json'
out_path = '../../../../datasets/HateSPic/MMHS/lstm_data/lstm_data_50k_3workers_regression_balanced/'

out_file_train = open(out_path + 'tweets.train', 'w')
out_file_val = open(out_path + 'tweets.val', 'w')
out_file_test = open(out_path + 'tweets.test', 'w')

print("Loading data ...")
data = json.load(open(data_path,'r'))

train = []
val = []
test = []

print("Read classifications splits to balance data")
class_train_ids = []
class_val_ids = []
class_test_ids = []
for l in open('../../../../datasets/HateSPic/MMHS/lstm_data/lstm_data_50k_3workers_classification/tweets.train_hate'):
    class_train_ids.append(int(l.split(',')[0]))
for l in open('../../../../datasets/HateSPic/MMHS/lstm_data/lstm_data_50k_3workers_classification/tweets.train_nothate'):
    class_train_ids.append(int(l.split(',')[0]))
for l in open('../../../../datasets/HateSPic/MMHS/lstm_data/lstm_data_50k_3workers_classification/tweets.val_hate'):
    class_val_ids.append(int(l.split(',')[0]))
for l in open('../../../../datasets/HateSPic/MMHS/lstm_data/lstm_data_50k_3workers_classification/tweets.val_nothate'):
    class_val_ids.append(int(l.split(',')[0]))
for l in open('../../../../datasets/HateSPic/MMHS/lstm_data/lstm_data_50k_3workers_classification/tweets.test_all'):
    class_test_ids.append(int(l.split(',')[0]))


print("Generating lstm data")
for k,v in data.iteritems():
    total_hate = 0
    for label in v['labels']:
        if label > 0:
            total_hate += 1

    label = total_hate / 3.0

    text = tweet_preprocessing(v['tweet_text'].encode('utf-8'))

    out_line = str(k) + ',' + text + ',' + str(label) + '\n'

    if int(k) in class_train_ids:  out_file_train.write(out_line)
    if int(k) in class_val_ids:  out_file_val.write(out_line)
    if int(k) in class_test_ids:  out_file_test.write(out_line)


print("DONE")