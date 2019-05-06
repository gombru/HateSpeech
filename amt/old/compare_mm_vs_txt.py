import cv2
import json
import numpy as np

results_file_txt = '../../../datasets/HateSPic/AMT/results/3rd_test_text.csv'
results_file_mm = '../../../datasets/HateSPic/AMT/results/3rd_test_mm.csv'
file_ids_txt_notHat_mm_hate_out = open('../../../datasets/HateSPic/AMT/results/3rdTest_ids_txt_Hate_mm_notHate.txt','w')

results_txt = {}
results_mm = {}

for i,line in enumerate(open(results_file_txt,'r')):
    if i == 0: continue
    data = line.split(',')
    tweet_id = int(data[-2].split('/')[-1].strip('"').strip('\n').strip('\t'))
    label = 1
    if 'No' in data[-1].strip('\n').strip('\t').strip('\u2019'):
        label = 0
    results_txt[tweet_id] = label

for i,line in enumerate(open(results_file_mm,'r')):
    if i == 0: continue
    data = line.split(',')
    tweet_id = int(data[-2].split('/')[-1].strip('"').strip('\n').strip('\t'))
    label = 1
    label_str = data[-1].strip('\n').strip('\t').strip('\u2019')
    if 'Not' in label_str:
    # if 'Not' in label_str or 'Other' in label_str:
        label = 0
    results_mm[tweet_id] = label

txt_hate = 0
txt_notHate = 0
mm_hate = 0
mm_notHate = 0
txt_hate_mm_hate = 0
txt_notHat_mm_hate = 0
ids=[]
for k,v in results_txt.iteritems():
    if v == 0:
        txt_notHate+=1
        if results_mm[k] == 1:
            txt_notHat_mm_hate += 1
            # ids.append(k)

    else:
        txt_hate+=1
        if results_mm[k] == 1:
            txt_hate_mm_hate+=1
        else:
            ids.append(k)

for k,v in results_mm.iteritems():
    if v == 0:
        mm_notHate+=1
    else:
        mm_hate+=1


print("Txt hate: " + str(txt_hate) + " Txt Not Hate " + str(txt_notHate) + " Txt_hate_mm_hate " + str(txt_hate_mm_hate) + " Txt_not_Hate_mm_hate " + str(txt_notHat_mm_hate))
print("MM hate: " + str(mm_hate) + " MM Not Hate " + str(mm_notHate))

for id in ids:
    file_ids_txt_notHat_mm_hate_out.write(str(id)+'\n')