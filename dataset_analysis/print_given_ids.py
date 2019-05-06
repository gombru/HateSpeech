import cv2
import json
import numpy as np
import os

ids_file = '../../../datasets/HateSPic//MMHS/anns/analysis/ids_notHate.txt'
out_folder = '../../../datasets/HateSPic/MMHS/anns/analysis/ids_notHate/'

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

ids=[]
for i,line in enumerate(open(ids_file,'r')):
    ids.append(int(line))
    # ids.append((line.split(',')))

for tweet_id in ids:
    # final_label = int(tweet_id[1])
    # tweet_id = int(tweet_id[0])
    # try:
    im = cv2.imread('../../../datasets/HateSPic/MMHS/img_resized/' + str(tweet_id) + '.jpg', 1)
    height, width, channels = im.shape
    im = cv2.resize(im, (500, int(500*(height/float(width)))))
    height, width, channels = im.shape
    out_image = np.ones((height + 115, width, 3), np.uint8)
    out_image = out_image * 255
    out_image[40:height+40,0:width,:] = im
    text = json.load(open('../../../datasets/HateSPic/MMHS/json/' + str(tweet_id) + '.json'))['text'].strip('\n').strip('\t').encode("utf8")
    print(text)
    # print("json loaded")
    font = cv2.FONT_HERSHEY_SIMPLEX
    texts = []
    if len(text) > width*0.1:
        for i in range(0,int((len(text) /(width*0.1)))+1):
            # print(str(int((len(text) /(width*0.1)))) + " lines")
            cv2.putText(out_image, text[int(i*width*0.1):int((i+1)*width*0.1)], (10, height + 60 + i*20), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        cv2.putText(out_image, text, (10, height + 60), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


    # c = (20, 20, 255)
    # text = 'Only Text: Hate'
    # cv2.putText(out_image, text.replace('\n', ' '), (6+130 * 0, 25), font, 0.6, c, 2, cv2.LINE_AA)
    # c = (20, 255, 20)
    # text = 'MM Tweet: Not Hate'
    # cv2.putText(out_image, text.replace('\n', ' '), (6 + 130 * 2, 25), font, 0.6, c, 2, cv2.LINE_AA)


    # if final_label == 1:
    #     c = (20, 20, 255)
    #     cv2.putText(out_image, "Misclassified to Hate", (6 + 130 * 2, 25), font, 0.6, c, 2, cv2.LINE_AA)
    # else:
    #     c = (20, 255, 20)
    #     cv2.putText(out_image, "Misclassified to Not Hate", (6 + 130 * 2, 25), font, 0.6, c, 2, cv2.LINE_AA)


    cv2.imwrite(out_folder + str(tweet_id) + '.jpg', out_image)
    # cv2.imwrite('../../../datasets/HateSPic/HateSPic/wrong_ids/' + str(tweet_id) + '.jpg', out_image)

        # print("Saved")
    # except:
    #     print("Error")