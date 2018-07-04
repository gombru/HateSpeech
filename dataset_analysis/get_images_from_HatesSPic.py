# Get the images labeled from HateSPic labeler to the same folder

import os
import json
from shutil import copyfile

generated_base_path = '../../../datasets/HateSPic/HateSPicLabeler/generated_json/'
datasets = ['HateSPic/','SemiSupervised/','WZ-LS/']
base_path = '../../../datasets/HateSPic/'

hate = {}
nothate = {}
total_votes = 0
total_tweets = 0
hate_count = 0
nothate_count = 0

for i,dataset in enumerate(datasets):
    for file in os.listdir(generated_base_path + dataset):
        id = json.load(open(generated_base_path + dataset + file, 'r'))['id']
        if i == 0:
            copyfile(base_path + 'twitter/' + 'img/' + str(id) + '.jpg', base_path + 'HateSPic/img/' + str(id) + '.jpg')
        if i == 1:
            copyfile(base_path + 'hate_speech_icwsm18/' + 'img/' + str(id) + '.jpg', base_path + 'HateSPic/img/' + str(id) + '.jpg')
        if i == 2:
            copyfile(base_path + 'Zhang/wz-ls/' + 'img/' + str(id) + '.jpg', base_path + 'HateSPic/img/' + str(id) + '.jpg')

print "Done"