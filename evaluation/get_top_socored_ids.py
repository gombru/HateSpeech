import numpy as np
import operator
import shutil

model_name = 'FCM_I_ADAM_bs32_lrMMe6_lrCNNe7_epoch_130_ValAcc_62'

results = {}
with open('../../../datasets/HateSPic/HateSPic/results/' + model_name + '/test.txt') as f:
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
    shutil.copyfile('../../../datasets/HateSPic/HateSPic/img_resized/' + str(str(r[0])) + '.jpg', '../../../datasets/HateSPic/HateSPic/top_scored_imgs/' + str(i) + '-' + str(r[0]) + '.jpg')

print("Done")