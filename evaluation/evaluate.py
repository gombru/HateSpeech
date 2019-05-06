import numpy as np
import matplotlib.pyplot as plt
import json

def evaluate(results, model_name):


    thresholds = np.arange(0, 1, 0.001)
    # thresholds = np.arange(0.4, 0.44, 0.0000001)


    best_f = 0
    best_th = 0
    best_f_re = 0
    best_f_pr = 0

    best_accuracy = 0
    acc_hate_best_accuracy = 0
    acc_notHate_best_accuracy = 0
    best_acc_th = 0

    precisions = []
    recalls = []
    fpr = []

    for th in thresholds:
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        cur_wrong_ids = []
        cur_correct_ids = []

        for r in results:
            if r[0] == 1 and r[1] >= th:
                tp += 1
                cur_correct_ids.append([r[2], 1])
            elif r[0] == 1 and r[1] < th:
                fn += 1
                cur_wrong_ids.append([r[2], 0])
            elif r[0] == 0 and r[1] < th:
                tn += 1
                cur_correct_ids.append([r[2], 0])
            elif r[0] == 0 and r[1] >= th:
                fp += 1
                cur_wrong_ids.append([r[2], 1])

        if tp > 0:
            pr = tp / float((tp + fp))
            re = tp / float((tp + fn))

        if pr + re > 0:
            f = 2 * (pr * re) / (pr + re)
        else:
            f = 0

        precisions.append(pr)
        recalls.append(re)
        fpr.append(tn / float(tn + fp))

        accuracy_hate = re
        if tn + fn > 0:
            accuracy_notHate = tn / float(tn + fp)
        else:
            accuracy_notHate = 0
        accuracy = (accuracy_hate + accuracy_notHate) / 2

        if accuracy > best_accuracy:
            wrong_ids = cur_wrong_ids
            correct_ids = cur_correct_ids
            best_accuracy = accuracy
            acc_hate_best_accuracy = accuracy_hate
            acc_notHate_best_accuracy = accuracy_notHate
            best_acc_th = th

        if f > best_f:
            best_f = f
            best_th = th
            best_f_pr = pr
            best_f_re = re

        print("thr " + str(th) + " --> F1: " + str(f) + " PR: " + str(pr) + " RE: " + str(re) + " ACC Hate: " + str(
            accuracy_hate) + " ACC NotHate: " + str(accuracy_notHate) + " ACC mean: " + str(accuracy))

    print("Best F1:  thr " + str(best_th) + " --> F1: " + str(best_f) + " PR: " + str(best_f_pr) + " RE: " + str(best_f_re))
    print("Best mean ACC:  thr " + str(best_acc_th) + " --> ACC: " + str(best_accuracy*100) + " Hate ACC: " + str(
        acc_hate_best_accuracy*100) + " Not Hate ACC: " + str(acc_notHate_best_accuracy*100))

    # Plot P-R
    # plt.plot(recalls, precisions)
    # plt.ylabel('Precision')
    # plt.xlabel('Recall')
    # plt.title(model_name)
    # plt.ylim(0, 1)
    # plt.xlim(0, 1)
    # plt.show()

    # Print ROC curve
    # plt.plot(recalls, fpr)
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.title("ROC " + model_name)
    # plt.show()

    # Print AUC
    auc = np.trapz(recalls, fpr)
    print('AUC:' + str(auc))


    save_data = {}
    save_data['precisions'] = precisions
    save_data['recalls'] = recalls
    save_data['fpr'] = fpr

    with open('../../../datasets/HateSPic/MMHS/evaluation_results/' + model_name + '.json', 'w') as outfile:
        json.dump(save_data, outfile)

    with open('../../../datasets/HateSPic/MMHS/evaluation_results/' + model_name + '_wrong_ids.txt', 'w') as outfile:
        for id in wrong_ids:
            outfile.write(str(id[0]) + ',' + str(id[1]) + '\n')

    with open('../../../datasets/HateSPic/MMHS/evaluation_results/' + model_name + '_correct_ids.txt', 'w') as outfile:
        for id in correct_ids:
            outfile.write(str(id[0]) + ',' + str(id[1]) + '\n')





