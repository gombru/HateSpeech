from preprocess_tweets import tweet_preprocessing
import json
import random

# Read terms to filter out tweets
# terms2dicard = []
# for line in open('../../../../datasets/HateSPic/AMT/50K/discardedTerms_tweetsPerTerm_50k.txt','r'):
#     term = line.split('-')[0].lower().strip(' ')
#     terms2dicard.append(term)

selected_terms = ['nigga','nigger']

data_path = '../../../../datasets/HateSPic/MMHS50K/anns/MMHS50K_GT.json'
out_path = '../../../../datasets/HateSPic/MMHS50K/lstm_data_niggaNigger/'

out_file_train_hate = open(out_path + 'tweets.train_hate', 'w')
out_file_val_hate = open(out_path + 'tweets.val_hate', 'w')
out_file_test_hate = open(out_path + 'tweets.test_hate', 'w')

out_file_train_nothate = open(out_path + 'tweets.train_nothate', 'w')
out_file_val_nothate = open(out_path + 'tweets.val_nothate', 'w')
out_file_test_nothate = open(out_path + 'tweets.test_nothate', 'w')

print("Loading data ...")
data = json.load(open(data_path,'r'))

train_hate = []
train_nothate = []
val_hate = []
val_nothate = []
test_hate = []
test_nothate = []

dicarded = 0
discarded_by_terms = 0
print("Generating lstm data")
for k,v in data.iteritems():

    label = 0
    # Discard Other Hate
    if v['label'] == 5:
        dicarded += 1
        continue
    if v['label'] > 0:
        label = 1 # Any type of hate
    text = tweet_preprocessing(v['tweet_text'].encode('utf-8'))

    # Discard if containing a term to dicard
    # discard = False
    # text_lower = text.lower()
    # for term in terms2dicard:
    #     if term in text_lower:
    #         discard = True
    #         discarded_by_terms +=1
    #         break
    # if discard: continue

    # Discard if not containing a selected term
    discard = True
    for term in selected_terms:
        if term in text.lower():
            discard = False
            break
    if discard:
        discarded_by_terms+=1
        continue

    split_selector = random.randint(1,10)

    if label == 1:
        if split_selector > 8:
            val_hate.append(str(k) + ',' + text + '\n')
        elif split_selector > 7:
            test_hate.append(str(k) + ',' + text + '\n')
        else:
            train_hate.append(str(k) + ',' + text + '\n')

    else:
        if split_selector > 8:
            val_nothate.append(str(k) + ',' + text + '\n')
        elif split_selector > 7:
            test_nothate.append(str(k) + ',' + text + '\n')
        else:
            train_nothate.append(str(k) + ',' + text + '\n')

print("Writing balanced LSTM data")
val_nothate_reduced = val_nothate[:len(val_hate)]
test_nothate_reduced = test_nothate[:len(test_hate)]
train_nothate = train_nothate + val_nothate[len(val_hate):] + val_nothate[len(val_hate):]

for l in train_hate: out_file_train_hate.write(l)
for l in train_nothate: out_file_train_nothate.write(l)
for l in val_hate: out_file_val_hate.write(l)
for l in val_nothate_reduced: out_file_val_nothate.write(l)
for l in test_hate: out_file_test_hate.write(l)
for l in test_nothate_reduced: out_file_test_nothate.write(l)

print("Discarded by OtherHate: " + str(dicarded))
print("Discarded by Terms: " + str(discarded_by_terms))

print "DONE"