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

num_in_out_folder = 100000

white_img_th = 150 # A less permisive text thereshold is applied to them
extreme_white_img_th = 200 # Dicard all images over this whiteness
white_img_txt_th = 0.2
txt_th = 0.3

#Compute heatmaps from images in txt
img_dir = '../../../datasets/HateSPic/MMHS/img_extra/'
json_dir = '../../../datasets/HateSPic/MMHS/json_extra/'
out_json_dir = '../../../datasets/HateSPic/AMT/MMHS2/2label_extra/'
out_dir = '../../../datasets/HateSPic/twitter/txt_img_fitlered/'
discarded_dir = '../../../datasets/HateSPic/twitter/txt_img_discarded/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
if not os.path.exists(discarded_dir):
    os.makedirs(discarded_dir)
if not os.path.exists(out_json_dir):
    os.makedirs(out_json_dir)


img_paths = []
for file in os.listdir(json_dir): img_paths.append(img_dir + file.replace('.json','.jpg'))

# load net
net = caffe.Net('deploy.prototxt', '../../../datasets/COCO-Text/fcn8s-atonce.caffemodel', caffe.TEST)


print 'Filtering ...'

count = 0
start = time.time()
dicardedByLSTM = 0
for img_path in img_paths:

    if os.path.exists(out_dir + img_path.split('/')[-1]) or os.path.exists(discarded_dir + img_path.split('/')[-1]):
        print("File exists, skipping")
        continue

    path, dirs, files = next(os.walk(out_json_dir))
    file_count = len(files)
    print("Num files in dest dir: " + str(file_count))
    if file_count == num_in_out_folder:
        print("File count reached, breaking")
        break



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

print("Discarded by LSTM " + str(dicardedByLSTM) + " from " + str(len(img_paths)))
print("DONE")