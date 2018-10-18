import cv2
import json
import numpy as np

results_file_txt = '../../../datasets/HateSPic/HateSPic/AMT/results/1st_test.csv'
results_file_mm = '../../../datasets/HateSPic/HateSPic/AMT/results/1st_test.csv'

results_txt = {}
results_mm = {}

for i,line in enumerate(open(results_file_txt,'r')):
    if i == 0: continue
    data = line.split(',')
    tweet_id = int(data[-2].split('/')[-1].strip('"').strip('\n').strip('\t'))
    if 'no' in data[-1].strip('\n').strip('\t').strip('\u2019'): label = 0
    results_txt[tweet_id] = label

for i,line in enumerate(open(results_file_mm,'r')):
    if i == 0: continue
    data = line.split(',')
    tweet_id = int(data[-2].split('/')[-1].strip('"').strip('\n').strip('\t'))
    if 'no' in data[-1].strip('\n').strip('\t').strip('\u2019'): label = 0
    results_mm[tweet_id] = label

txt_hate = 0
txt_notHate = 0
txt_hate_mm_hate = 0
txt_notHat_mm_hate = 0
for k,v in results_txt:
    if v == 0:
        txt_notHate+=1
        if results_mm[k] == 1:
            txt_notHat_mm_hate += 1
    else:
        txt_hate+=1
        if results_mm[k] == 1:

