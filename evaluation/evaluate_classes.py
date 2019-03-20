import numpy as np

def compute_softmax_max(scores):
    softmax_scores = np.zeros(len(scores))
    exp_scores = np.exp(scores)
    exp_scores_sum = sum(exp_scores)
    for i,e_s in enumerate(exp_scores):
        softmax_scores[i] = e_s / exp_scores_sum
    return np.argmax(softmax_scores)


def evaluate_classes(results):
    class_labels = ['NotHate', 'Racist', 'Sexist', 'Homophobe']
    tp_classes = np.zeros(len(class_labels))
    fn_classes = np.zeros(len(class_labels))
    accuracies = np.zeros(len(class_labels))

    for r in results:
        inferred_class = compute_softmax_max(r[1:-1])
        if inferred_class == r[0]: tp_classes[r[0]] += 1
        else: fn_classes[r[0]] +=1

    for i,cl in enumerate(class_labels):
        accuracies[i] = tp_classes[i] / float((tp_classes[i] + fn_classes[i]))
        print("Acc class " + cl + ": " + str(accuracies[i]))

    print("Mean ACC: " + str(accuracies.mean()))







