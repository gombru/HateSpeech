import os
import string
import json

base_path = '../../datasets/instaEmotions/json_filtered/'
out_path = '../../datasets/instaEmotions/lstm_data/'
emotion_dirs = ['amusement breathtaking thrilling','anger angry wrath rage','awe admiration astonishment','contentment complacency diversion pleasure','disgusting abominable awful disgusting distasteful hateful','exciting stimulating','fear scare scary terror','sadness anguish dolor grief melancholy']
base_emotions = ['amusement','anger','awe','contentment','disgust','excitement','fear','sadness']
whitelist = string.letters + string.digits + ' '

for i, emotion in enumerate(base_emotions):
    print emotion
    text = ""
    cur_emotion_dirs = emotion_dirs[i].split(' ')
    for emotion_dir in cur_emotion_dirs:
        print emotion + " --> " + emotion_dir
        for file in os.listdir(base_path + emotion_dir):
            # try:
            with open(base_path + emotion_dir + '/' + file,'r') as f:
                filtered_caption = ""
                data = json.load(f)
                caption = data['caption']
                for l in caption:
                    caption = l.rstrip()
                    caption = caption.replace('#', ' ')
                    for char in caption:
                        if char in whitelist:
                            filtered_caption += char
                    filtered_caption = filtered_caption.decode('utf-8').lower()
            # except:
            #     print "Error getting caption, continuing"
            #     continue

            text += filtered_caption

            text += ' . \n'
    with open(out_path + 'captions.' + emotion, 'w') as f:
        f.write(text)

print "DONE"