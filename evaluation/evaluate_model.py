from evaluate import evaluate
import numpy as np
import random

def run_evaluation(model, mode):

    sum_hate = 0
    sum_nothate = 0

    model = model.strip('.pth')
    results = []
    with open('../../../datasets/HateSPic/MMHS/results/' + model + '/test.txt') as f:
        for line in f:
            data = line.split(',')
            label = int(data[1])
            if mode == "classification":
                hate_score = float(data[3])
                notHate_score = float(data[2])
                softmax_hate_score = np.exp(hate_score) / (np.exp(hate_score) + np.exp(notHate_score))
            elif mode == "regression":
                softmax_hate_score = float(data[2])
            # softmax_hate_score = random.random()
            # softmax_hate_score = random.random()
            # model_name = "Random"
            if label == 0: sum_nothate+=softmax_hate_score
            if label == 1: sum_hate+=softmax_hate_score
            results.append([label, softmax_hate_score, int(data[0])])

    print(sum_hate,sum_nothate)
    evaluate(results, model)
#
# model_name = 'MMHS_regression_FCM_TT_epoch_3_ValLoss_0.08'
# run_evaluation(model_name,'regression')