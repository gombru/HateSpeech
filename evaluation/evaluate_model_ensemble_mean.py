from evaluate import evaluate
import random
import numpy as np

model_name = 'MMHSv2mm_SCM_ALL_ADAM_bs32_lrMMe6_lrCNNe7_NoCNNInit_epoch_68_ValAcc_80'

visual_model_max_score = {}
visual_model_data = {}
with open('../../../datasets/HateSPic/HateSPic/results/' + model_name + '/test.txt') as f:
    for line in f:
        data = line.split(',')
        label = int(data[1])
        hate_score = float(data[3])
        notHate_score = float(data[2])
        softmax_hate_score = np.exp(hate_score) / (np.exp(hate_score) + np.exp(notHate_score))
        softmax_not_hate_score = np.exp(notHate_score) / (np.exp(hate_score) + np.exp(notHate_score))
        visual_model_max_score[int(data[0])] = max(softmax_hate_score,softmax_not_hate_score)
        visual_model_data[int(data[0])] = [label, softmax_hate_score, int(data[0])]

model_name = 'MMHS-v2mm_FCN_TT_ADAM_bs32_lrMMe6_lrCNNe7_epoch_11_ValAcc_79'

by_txt = 0
by_visual = 0
results = []
with open('../../../datasets/HateSPic/HateSPic/results/' + model_name + '/test.txt') as f:
    for line in f:
        data = line.split(',')
        label = int(data[1])
        hate_score = float(data[3])
        notHate_score = float(data[2])
        softmax_hate_score = np.exp(hate_score) / (np.exp(hate_score) + np.exp(notHate_score))
        softmax_not_hate_score = np.exp(notHate_score) / (np.exp(hate_score) + np.exp(notHate_score))
        textual_max_score = max(softmax_hate_score,softmax_not_hate_score)
        if textual_max_score > visual_model_max_score[int(data[0])]:
            results.append([label, softmax_hate_score, int(data[0])])
            by_txt += 1
        else:
            results.append(visual_model_data[int(data[0])])
            by_visual+=1

model_name = "Ensemble"
random.shuffle(results)
evaluate(results, model_name)

print("Scored by visual model: " + str(by_visual) + " / Scored by textual model: " + str(by_txt))