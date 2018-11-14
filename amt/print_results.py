import cv2
import json
import numpy as np

results_file = '../../../datasets/HateSPic/HateSPic/AMT/results/1st_test.csv'
results = {}

for i,line in enumerate(open(results_file,'r')):
    if i == 0: continue
    data = line.split(',')
    tweet_id = int(data[-2].split('/')[-1].strip('"').strip('\n').strip('\t'))
    label = data[-1].strip('\n').strip('\t').strip('\u2019')
    if tweet_id in results: results[tweet_id].append(label)
    else: results[tweet_id] = [label]

for tweet_id,labels in results.iteritems():
    # try:
    im = cv2.imread('../../../datasets/HateSPic/HateSPic/img_resized/' + str(tweet_id) + '.jpg', 1)
    height, width, channels = im.shape
    im = cv2.resize(im, (500, int(500*(height/float(width)))))
    height, width, channels = im.shape
    out_image = np.ones((height + 115, width, 3), np.uint8)
    out_image = out_image * 255
    out_image[40:height+40,0:width,:] = im
    text = json.load(open('../../../datasets/HateSPic/twitter/json_all/' + str(tweet_id) + '.json'))['text'].strip('\n').strip('\t').encode("utf8")
    print text
    # print("json loaded")
    font = cv2.FONT_HERSHEY_SIMPLEX
    texts = []
    if len(text) > width*0.1:
        for i in range(0,int((len(text) /(width*0.1)))+1):
            # print(str(int((len(text) /(width*0.1)))) + " lines")
            cv2.putText(out_image, text[int(i*width*0.1):int((i+1)*width*0.1)], (10, height + 60 + i*20), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        cv2.putText(out_image, text, (10, height + 60), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    labels_text = ''
    for i,l in enumerate(labels):
        if 'NoHate' in l:
            c = (20, 255, 20)
            text = ' NotHate '
        if 'TxtHate' in l:
            c = (20, 20, 255)
            text = 'HateInText '
        if 'MMHate' in l:
            c = (20, 20, 255)
            text = 'HateIfImg '

        cv2.putText(out_image, text.replace('\n', ' '), (6+130 * i, 25), font, 0.6, c, 2, cv2.LINE_AA)

    cv2.imwrite('../../../datasets/HateSPic/HateSPic/AMT/img_results/' + str(tweet_id) + '.jpg', out_image)
    # print("Saved")
    # except:
    #     print("Error")