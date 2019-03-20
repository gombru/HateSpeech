import numpy as np
import operator
import shutil
import os

model_name = 'MMHS50K_niggaNigger_FCM_TT_epoch_71_ValAcc_56'
out_folder_name = 'topIMG_FCM_TT_niggaNigger_top_hate'
out_file = open('../../../datasets/HateSPic/MMHS50K/top_scored/' + out_folder_name + '.txt','w')

if not os.path.exists('../../../datasets/HateSPic/MMHS50K/top_scored/' + out_folder_name):
    os.makedirs('../../../datasets/HateSPic/MMHS50K/top_scored/' + out_folder_name)

results = {}
with open('../../../datasets/HateSPic/MMHS50K/results/' + model_name + '/test.txt') as f:
    for line in f:
        data = line.split(',')
        id = int(data[0])
        label = int(data[1])
        hate_score = float(data[3])
        notHate_score = float(data[2])
        softmax_hate_score = np.exp(hate_score) / (np.exp(hate_score) + np.exp(notHate_score))
        results[id] = softmax_hate_score


results = sorted(results.items(), key=operator.itemgetter(1))
results = list(reversed(results))


for i,r in enumerate(results):
    if i == 50: break
    print r[1]
    shutil.copyfile('../../../datasets/HateSPic/MMHS50K/img_resized/' + str(str(r[0])) + '.jpg', '../../../datasets/HateSPic/MMHS50K/top_scored/' + out_folder_name + '/' + str(i) + '-' + str(r[0]) + '.jpg')
    out_file.write(str(r[0]) + '\n')
out_file.close()
print("Done")