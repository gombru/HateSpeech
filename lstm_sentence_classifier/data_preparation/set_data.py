import os
import string

base_path = '../../datasets/EmotionDataset/txt/'
out_path = '../../datasets/EmotionDataset/lstm_data/filtered/'
emotions = ['amusement','anger','awe','contentment','disgust','excitement','fear','sadness']
whitelist = string.letters + string.digits + ' '

filtered_gt_data = []
fitlered_gt_path = '../../datasets/EmotionDataset/gt_filtered/test.txt'
filtered_gt = open(fitlered_gt_path, 'r')
for f in filtered_gt:
    filtered_gt_data.append(f.split(',')[0].split('/')[1][:-4])

# USE THIS TO BUILD THE CLEAN DATA
clean_data = []
path = '../../datasets/EmotionDataset/gt_filtered/train.txt'
data = open(path, 'r')
for f in data:
    clean_data.append(f.split(',')[0].split('/')[1][:-4])
path = '../../datasets/EmotionDataset/gt_filtered/val.txt'
data = open(path, 'r')
for f in data:
    clean_data.append(f.split(',')[0].split('/')[1][:-4])

for emotion in emotions:
    print emotion
    text = ""
    for file in os.listdir(base_path + emotion):

        # USE THIS TO BUILD THE CLEAN DATA
        if file[:-4] not in clean_data: continue

        # USE THIS TO BUILD / FILTER TEST SET
        if file[:-4] in filtered_gt_data:
            print "Skiping test instance"
            continue

        with open(base_path + emotion + '/' + file,'r') as f:

            for l in f:
                caption = l.rstrip()
                caption = caption.replace('#', ' ')
                filtered_caption = ""
                for char in caption:
                    if char in whitelist:
                        filtered_caption += char
                filtered_caption = filtered_caption.decode('utf-8').lower()

        text += file[:-4] + ',' + filtered_caption

        text += ' . \n'
    with open(out_path + 'captions.' + emotion, 'w') as f:
        f.write(text)

print "DONE"