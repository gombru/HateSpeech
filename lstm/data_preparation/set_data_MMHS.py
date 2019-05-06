from preprocess_tweets import tweet_preprocessing
import json
import random


data_path = '../../../../datasets/HateSPic/MMHS/anns/MMHS150K_GT.json'
out_path = '../../../../datasets/HateSPic/MMHS/lstm_data/lstm_data_150k_classification/'

out_file_train_hate = open(out_path + 'tweets.train_hate', 'w')
out_file_val_hate = open(out_path + 'tweets.val_hate', 'w')
out_file_test_hate = open(out_path + 'tweets.test_hate', 'w')

out_file_train_nothate = open(out_path + 'tweets.train_nothate', 'w')
out_file_val_nothate = open(out_path + 'tweets.val_nothate', 'w')
out_file_test_nothate = open(out_path + 'tweets.test_nothate', 'w')

print("Loading data ...")
data = json.load(open(data_path,'r'))
print("Randomizing dict order")
data_list = list(data.items())
random.shuffle(data_list)
data = dict(data_list)

train_hate = []
train_nothate = []
val_hate = []
val_nothate = []
test_hate = []
test_nothate = []

num_test = 10000
num_val = 5000

discarded = 0

count_hate = 0
count_nothate = 0

print("Generating lstm data")
for k,v in data.iteritems():
    total_hate = 0
    for label in v['labels']:
        if label > 0:
            total_hate += 1

    label = float(total_hate) / 3.0

    if label > 0.5: label = 1
    else: label = 0

    text = tweet_preprocessing(v['tweet_text'].encode('utf-8'))

    # split_selector = random.randint(1,10)

    if label == 0:
        if count_nothate < 2500:
            val_nothate.append(str(k) + ',' + text + '\n')
        elif count_nothate < 7500:
            test_nothate.append(str(k) + ',' + text + '\n')
        else:
            train_nothate.append(str(k) + ',' + text + '\n')
        count_nothate+=1

    else:
        if count_hate < 2500:
            val_hate.append(str(k) + ',' + text + '\n')
        elif count_hate < 7500:
            test_hate.append(str(k) + ',' + text + '\n')
        else:
            train_hate.append(str(k) + ',' + text + '\n')
        count_hate+=1

print("Val size")
print(len(val_hate))
print(len(val_nothate))
print("Test size")
print(len(test_hate))
print(len(test_nothate))
print("Train size")
print(len(train_hate))
print(len(train_nothate))

print("Writing  data")
# val_nothate_reduced = val_nothate[:len(val_hate)]
# test_nothate_reduced = test_nothate[:len(test_hate)]
# train_nothate = train_nothate + val_nothate[len(val_hate):] + val_nothate[len(val_hate):]

for l in train_hate: out_file_train_hate.write(l)
for l in train_nothate: out_file_train_nothate.write(l)
for l in val_hate: out_file_val_hate.write(l)
for l in val_nothate: out_file_val_nothate.write(l)
for l in test_hate: out_file_test_hate.write(l)
for l in test_nothate: out_file_test_nothate.write(l)

print("DONE")