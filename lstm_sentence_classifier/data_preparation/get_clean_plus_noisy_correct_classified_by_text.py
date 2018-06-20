# This is used to filter the data that I'll use to train the 2 headed CNN
# The idea is, from the noisy dataset, keep those isntances that are in the clean dataset and those instances that
# are correctly classified by text

# Original files
original_train = open("../../datasets/EmotionDataset/lstm_gt/train_noisy.txt",'r')
original_val = open("../../datasets/EmotionDataset/lstm_gt/val_noisy.txt",'r')

# Out files
out_train = open("../../datasets/EmotionDataset/lstm_gt/train_noisy_cleaned.txt",'w')
out_val = open("../../datasets/EmotionDataset/lstm_gt/val_noisy_cleaned.txt",'w')

# Get filtered dataset ids
filtered_dataset_ids = []
data = open('../../datasets/EmotionDataset/gt_filtered/train.txt', 'r')
for f in data:
    filtered_dataset_ids.append(f.split(',')[0].split('/')[1][:-4])
data = open('../../datasets/EmotionDataset/gt_filtered/val.txt', 'r')
for f in data:
    filtered_dataset_ids.append(f.split(',')[0].split('/')[1][:-4])

# Get LSTM text classification result
correct_text_classification = {}
lstm_classification_train = open("../../datasets/EmotionDataset/lstm_gt/train_noisy_classification.txt",'r')
for el in lstm_classification_train:
    correct_text_classification[el.split(',')[0]] = int(el.split(',')[1])
lstm_classification_val = open("../../datasets/EmotionDataset/lstm_gt/val_noisy_classification.txt",'r')
for el in lstm_classification_val:
    correct_text_classification[el.split(',')[0]] = int(el.split(',')[1])


for el in original_train:
    save = False
    id = el.split(',')[0]
    if correct_text_classification[id] == 1: save = True
    if id in filtered_dataset_ids: save = True
    if save: out_train.write(el)

for el in original_val:
    save = False
    id = el.split(',')[0]
    if correct_text_classification[id] == 1: save = True
    if id in filtered_dataset_ids: save = True
    if save: out_val.write(el)

