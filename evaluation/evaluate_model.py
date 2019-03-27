from evaluate import evaluate
import numpy as np
import random

def run_evaluation(model):

    model = model.strip('.pth')
    results = []
    with open('../../../datasets/HateSPic/MMHS/results/' + model + '/test.txt') as f:
        for line in f:
            data = line.split(',')
            label = int(data[1])
            hate_score = float(data[3])
            notHate_score = float(data[2])
            softmax_hate_score = np.exp(hate_score) / (np.exp(hate_score) + np.exp(notHate_score))
            # softmax_hate_score = random.random()
            # model_name = "Random"
            results.append([label, softmax_hate_score, int(data[0])])
    evaluate(results, model)

# model_name = 'MMHS50K_noOther_FCM_DOHardEmbeddings095_ALL_ADAM_bs32_lrMMe6_lrCNNe7_epoch_165_ValAcc_65'
# run_evaluation(model_name)