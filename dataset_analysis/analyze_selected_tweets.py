# From the tweets with images I could retrieve from existing annotated datasets, find out how much are hate

import os
import json

base_path = '../../../datasets/HateSPic/HateSPicLabeler/filtered_original_json/WZ-LS/'

hate = 0
nothate =0

for file in os.listdir(base_path):
    i = json.load(open(base_path + file,'r'))
    if i['original_annotation'] == 1: hate += 1
    else: nothate += 1

print("Hate: " + str(hate) + " Not Hate " + str(nothate))