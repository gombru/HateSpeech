# Get the images labeled from HateSPic labeler to the same folder

import os
import json
from shutil import copyfile

generated_base_path = '../../../datasets/HateSPic/HateSPicLabeler/generated_json/'
datasets = ['HateSPic/','SemiSupervised/','WZ-LS/']
base_path = '../../../datasets/HateSPic/'
out_txt_file = open('../../../datasets/HateSPic/HateSPic/all_hate_txt.txt','w')

hate = {}
nothate = {}
total_votes = 0
total_tweets = 0
hate_count = 0
nothate_count = 0

for index,dataset in enumerate(datasets):
    for file in os.listdir(generated_base_path + dataset):
        id = json.load(open(generated_base_path + dataset + file, 'r'))['id']

        i = json.load(open(generated_base_path + dataset + file, 'r'))
        total_votes += i['hate_votes'] + i['not_hate_votes']
        if i['hate_votes'] > i['not_hate_votes']:

            if index == 0:
                copyfile(base_path + 'twitter/' + 'img/' + str(id) + '.jpg', base_path + 'HateSPic/hate_labeled_data/' + str(id) + '.jpg')
                copyfile(base_path + 'twitter/' + 'json/' + str(id) + '.json', base_path + 'HateSPic/hate_labeled_data/' + str(id) + '.json')
                out_txt_file.write(i['text'].encode("utf-8") + '\n')

            # if i == 1:
            #     copyfile(base_path + 'hate_speech_icwsm18/' + 'img/' + str(id) + '.jpg', base_path + 'HateSPic/img/' + str(id) + '.jpg')
            # if i == 2:
            #     copyfile(base_path + 'Zhang/wz-ls/' + 'img/' + str(id) + '.jpg', base_path + 'HateSPic/img/' + str(id) + '.jpg')

print "Done"