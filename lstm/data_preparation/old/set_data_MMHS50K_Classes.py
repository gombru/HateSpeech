from preprocess_tweets import tweet_preprocessing
import json
import random

data_path = '../../../../datasets/HateSPic/MMHS50K/anns/MMHS50K_GT.json'
out_path = '../../../../datasets/HateSPic/MMHS50K/lstm_data_classes/'

out_file_train = open(out_path + 'tweets.train', 'w')
out_file_val = open(out_path + 'tweets.val', 'w')
out_file_test = open(out_path + 'tweets.test', 'w')


print("Loading data ...")
data = json.load(open(data_path,'r'))

print("Generating lstm data")
for k,v in data.iteritems():

    # Discard "Other Hate" class
    if v['label'] == 5:
        continue
    # Discard "Religion Hate" class
    if v['label'] == 4:
        continue

    text = tweet_preprocessing(v['tweet_text'].encode('utf-8'))
    label_str = v['label_str'].encode('utf-8')

    split_selector = random.randint(1,10)

    if split_selector > 8:
        out_file_val.write(str(k) + ',' + label_str + ',' + text + '\n')
    elif split_selector > 7:
        out_file_test.write(str(k) + ',' + label_str + ',' + text + '\n')
    else:
        out_file_train.write(str(k) + ',' + label_str + ',' + text + '\n')


print "DONE"