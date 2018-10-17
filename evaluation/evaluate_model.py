from evaluate import evaluate
import numpy as np
import random


model_name = 'TKM-5-0-TextConcat_ALL_ADAM_bs32_lrMMe6_lrCNNe7_epoch_49_ValAcc_77'

results = []
with open('../../../datasets/HateSPic/HateSPic/results/' + model_name + '/test.txt') as f:
    for line in f:
        data = line.split(',')
        label = int(data[1])
        hate_score = float(data[3])
        notHate_score = float(data[2])
        softmax_hate_score = np.exp(hate_score) / (np.exp(hate_score) + np.exp(notHate_score))
        # softmax_hate_score = random.random()
        # model_name = "Random"
        results.append([label, softmax_hate_score])
evaluate(results, model_name)