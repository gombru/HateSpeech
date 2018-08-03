import sys
sys.path.append('/home/imatge/caffe-master/python')
import numpy as np
from PIL import Image
import caffe
import time
import os
from shutil import copyfile
import random

# Run in GPU
caffe.set_device(0)
caffe.set_mode_gpu()


twitter = True # If filtering fromt twitter also consider LSTM scores
lstm_filtering_probability = 5 # The probability that the LSTM filtering is not applied. I want to let in some "random" images
lstm_scores_path = '../../../datasets/HateSPic/twitter/lstm_scores.txt'
lstm_th = 0.5
white_img_th = 150 # A less permisive text thereshold is applied to them
extreme_white_img_th = 200 # Dicard all images over this whiteness
white_img_txt_th = 0.2
txt_th = 0.3

#Compute heatmaps from images in txt
img_dir = '../../../datasets/HateSPic/twitter/img/'
json_dir = '../../../datasets/HateSPic/twitter/json_2/'
out_json_dir = '../../../datasets/HateSPic/HateSPicLabeler/filtered_original_json/HateSPic/'
out_dir = '../../../datasets/HateSPic/twitter/txt_img_fitlered/'
discarded_dir = '../../../datasets/HateSPic/twitter/txt_img_discarded/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
if not os.path.exists(discarded_dir):
    os.makedirs(discarded_dir)
if not os.path.exists(out_json_dir):
    os.makedirs(out_json_dir)

# Load lstm data
if twitter:
    lstm_scores = {}
    for line in open(lstm_scores_path,'r'):
        lstm_scores[line.split(',')[0]] = float(line.split(',')[1])


img_paths = []
for file in os.listdir(json_dir): img_paths.append(img_dir + file.replace('.json','.jpg'))

# load net
net = caffe.Net('deploy.prototxt', '../../../datasets/COCO-Text/fcn8s-atonce.caffemodel', caffe.TEST)


print 'Filtering ...'

count = 0
start = time.time()

for img_path in img_paths:

    # LSTM discarding
    if twitter and random.randint(0,100) > lstm_filtering_probability:
        try:
            if lstm_scores[img_path.split('/')[-1].split('.')[0]] < lstm_th:
                print("Img discarded by LSTM")
                continue
        except:
            print("Couldn't find LSTM score")
            continue


    try:

        count = count + 1
        if count % 100 == 0:
            print count

        # load image
        im = Image.open(img_path)

        # Turn grayscale images to 3 channels
        if (im.size.__len__() == 2):
            im_gray = im
            im = Image.new("RGB", im_gray.size)
            im.paste(im_gray)

        #switch to BGR and substract mean
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= np.array((104.00698793,116.66876762,122.67891434))
        in_ = in_.transpose((2,0,1))

        # shape for input (data blob is N x C x H x W)
        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_

        # run net and take scores
        net.forward()

        # Compute SoftMax HeatMap
        hmap_0 = net.blobs['score_conv'].data[0][0, :, :]   #Text score
        hmap_1 = net.blobs['score_conv'].data[0][1, :, :]   #Backgroung score
        hmap_0 = np.exp(hmap_0)
        hmap_1 = np.exp(hmap_1)
        hmap_softmax = hmap_1 / (hmap_0 + hmap_1)

        txt_score = hmap_softmax.sum() / (im.size[0] * im.size[1])
        img_mean = np.array(im).sum() / (im.size[0] * im.size[1] * 3)


        # Threshold for white images (more probable to contain text)
        if txt_score < white_img_txt_th and img_mean > white_img_th and img_mean < extreme_white_img_th:
            copyfile(img_path, out_dir + img_path.split('/')[-1])
            copyfile(json_dir + img_path.split('/')[-1].split('.')[0] + '.json', out_json_dir + img_path.split('/')[-1].split('.')[0] + '.json')
            print("Image saved (using restrictive threshold because is white)")
        # Threshold for black images
        elif txt_score < txt_th and img_mean < white_img_th:
            copyfile(img_path, out_dir + img_path.split('/')[-1])
            copyfile(json_dir + img_path.split('/')[-1].split('.')[0] + '.json', out_json_dir + img_path.split('/')[-1].split('.')[0] + '.json')
            print("Image saved (using permissive threshold because is not white)")
        else:
            copyfile(img_path, discarded_dir + img_path.split('/')[-1])
            print("Image discarded by TextFCN")

        # im.show()
        # print "h"

    except:
        print("Error with image: " + img_path)

print("DONE")