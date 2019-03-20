from evaluate_classes import evaluate_classes
import random

def run_evaluation(model):

    model = model.strip('.pth')
    results = []
    with open('../../../datasets/HateSPic/MMHS50K/results/' + model + '/test.txt') as f:
        for line in f:
            d = line.split(',')
            results.append([int(d[1]), float(d[2]), float(d[3]), float(d[4]), float(d[6]), int(d[0])])
    evaluate_classes(results)

# model_name = 'MMHS50K_noOther_FCM_DOHardEmbeddings095_ALL_ADAM_bs32_lrMMe6_lrCNNe7_epoch_165_ValAcc_65'
# run_evaluation(model_name)