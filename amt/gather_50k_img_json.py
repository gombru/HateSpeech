# encoding=utf8
from shutil import  copyfile

base_path = '../../../datasets/HateSPic/AMT/50K/'

ids_file = open('../../../datasets/HateSPic/AMT/50K/ids_50000_amt.csv','r')

ids = []
for i,line in enumerate(ids_file):
    if i == 0: continue
    id = int(line.split('/')[-1].strip('\n'))
    ids.append(id)
print("Found ids: " + str(len(ids)))

for id in ids:
    copyfile('../../../datasets/HateSPic/AMT/50K/2label/' + str(id) + '.json', '../../../datasets/HateSPic/AMT/50K/json/' + str(id) + '.json')
    copyfile('../../../datasets/HateSPic/twitter/img/' + str(id) + '.jpg', '../../../datasets/HateSPic/AMT/50K/img/' + str(id) + '.jpg')


