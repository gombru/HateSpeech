from preprocess_tweets import tweet_preprocessing
import os
import json
import random

base_path = '../../../../datasets/HateSPic/HateSPicLabeler/generated_json_v2mm/'
datasets = ['HateSPic/','SemiSupervised/','WZ-LS/']
out_path = '../../../../datasets/HateSPic/lstm_data/HateSPic_nigga_faggot/'

out_file_train_hate = open(out_path + 'tweets.train_hate', 'w')
out_file_val_hate = open(out_path + 'tweets.val_hate', 'w')
out_file_test_hate = open(out_path + 'tweets.test_hate', 'w')

out_file_train_nothate = open(out_path + 'tweets.train_nothate', 'w')
out_file_val_nothate = open(out_path + 'tweets.val_nothate', 'w')
out_file_test_nothate = open(out_path + 'tweets.test_nothate', 'w')


for dataset in datasets:
    for file in os.listdir(base_path + dataset):
        info = json.load(open(base_path + dataset + file, 'r'))
        label = 0
        if info['hate_votes'] > info['not_hate_votes']:
            label = 1
        if 'mm_hate_votes' in info:
            if info['mm_hate_votes'] > info['not_hate_votes']:
                label = 1
        text = tweet_preprocessing(info['text'].encode('utf-8'))

        print("Selecting only 'nigga' and 'faggot' tweets")
        if 'nigga' not in text and 'faggot' not in text: continue
        # Discard short tweets
        # if len(text) < 5: continue
        # if len(text.split(' ')) < 3: continue

        split_selector = random.randint(0,19)

        if label == 1:
            if split_selector > 15:
                out_file_val_hate.write(str(info['id']) + ',' + text + '\n')
            elif split_selector > 11:
                out_file_test_hate.write(str(info['id']) + ',' + text + '\n')
            else:
                out_file_train_hate.write(str(info['id']) + ',' + text + '\n')

        else:
            if split_selector > 15:
                out_file_val_nothate.write(str(info['id']) + ',' + text + '\n')
            elif split_selector > 11:
                out_file_test_nothate.write(str(info['id']) + ',' + text + '\n')
            else:
                out_file_train_nothate.write(str(info['id']) + ',' + text + '\n')


print "BALANCE TEST DATA!!"
print "DONE"