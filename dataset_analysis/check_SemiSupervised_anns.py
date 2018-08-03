import os
import json

base_path = '../../../datasets/HateSPic/HateSPicLabeler/generated_json/SemiSupervised/'

hate = {}
nothate = {}
total_votes = 0
total_tweets = 0
hate_count = 0
nothate_count = 0

original_nonhate = 0
original_hate = 0

matched_hate = 0
matched_nothate = 0

for file in os.listdir(base_path):
    i = json.load(open(base_path + file, 'r'))
    total_votes += i['hate_votes'] + i['not_hate_votes']
    if i['original_annotation'] == 0: original_nonhate +=1
    if i['original_annotation'] == 1: original_hate +=1

    if i['hate_votes'] > i['not_hate_votes']:
        if i['original_annotation'] == 1: matched_hate += 1
        hate_count += 1
        if i['dataset'] in hate: hate[i['dataset']] += 1
        else: hate[i['dataset']] = 1
    else:
        if i['original_annotation'] == 0: matched_nothate += 1
        nothate_count += 1
        if i['dataset'] in nothate: nothate[i['dataset']] += 1
        else: nothate[i['dataset']] = 1
    total_tweets += 1

not_mached_anns = total_tweets - (matched_nothate + matched_hate)


print "Total tweets: " + str(total_tweets) + " Total votes: " + str(total_votes)
print "Total hate: " + str(hate_count) + " Total NotHate: " + str(nothate_count)

print "Original anns: Hate " + str(original_hate) + " Not Hate " + str(original_nonhate)
print "Matched anns " + str(matched_hate + matched_nothate) + " Not matched " + str(not_mached_anns) + " % matched " + str(100 * (matched_nothate + matched_hate) / total_tweets)
print "Acc hate " + str(100 * matched_hate / original_hate)
print "Acc not hate " + str(100 * matched_nothate / original_nonhate)