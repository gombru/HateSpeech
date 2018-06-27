import os
import json

base_path = '../../../datasets/HateSPic/HateSPicLabeler/generated_json/'
datasets = ['HateSPic/','SemiSupervised/','WZ-LS/']

hate = {}
nothate = {}
total_votes = 0
total_tweets = 0
hate_count = 0
nothate_count = 0

for dataset in datasets:
    for file in os.listdir(base_path + dataset):
        i = json.load(open(base_path + dataset + file, 'r'))
        total_votes += i['hate_votes'] + i['not_hate_votes']
        if i['hate_votes'] > i['not_hate_votes']:
            hate_count += 1
            if i['dataset'] in hate: hate[i['dataset']] += 1
            else: hate[i['dataset']] = 1
        else:
            nothate_count += 1
            if i['dataset'] in nothate: nothate[i['dataset']] += 1
            else: nothate[i['dataset']] = 1
        total_tweets += 1


print "Total tweets: " + str(total_tweets) + " Total votes: " + str(total_votes)
print "Total hate: " + str(hate_count) + " Total NotHate: " + str(nothate_count)


print "Hate"
print hate
print "Not Hate"
print nothate

