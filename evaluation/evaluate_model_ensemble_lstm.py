from evaluate import evaluate
import random
import numpy as np

model_name = 'MMHSv4mm_TKM-v2-CK-10-5-NoConcat_ALL_ADAM_bs32_lrMMe6_lrCNNe7_epoch_13_ValAcc_80'

visual_model_data = {}
with open('../../../datasets/HateSPic/HateSPic/results/' + model_name + '/test.txt') as f:
    for line in f:
        data = line.split(',')
        label = int(data[1])
        hate_score = float(data[3])
        notHate_score = float(data[2])
        softmax_hate_score = np.exp(hate_score) / (np.exp(hate_score) + np.exp(notHate_score))
        visual_model_data[int(data[0])] = softmax_hate_score

# model_name = 'MMHS-v2mm_FCN_TT_ADAM_bs32_lrMMe6_lrCNNe7_epoch_11_ValAcc_79'
#
# results = []
# with open('../../../datasets/HateSPic/HateSPic/results/' + model_name + '/test.txt') as f:
#     for line in f:
#         data = line.split(',')
#         label = int(data[1])
#         hate_score = float(data[3])
#         notHate_score = float(data[2])
#         softmax_hate_score = np.exp(hate_score) / (np.exp(hate_score) + np.exp(notHate_score))
#         softmax_hate_score = (softmax_hate_score + visual_model_data[int(data[0])]) / 2
#         # softmax_hate_score = max(softmax_hate_score, visual_model_data[int(data[0])])
#         results.append([label, softmax_hate_score, int(data[0])])
#

model_name = 'LSTM_MMHS10K-v4mm_tweet_text'
scores_file = 'MMHS10K-v4mm_lstm_scores' #'lstm_scores'

# Load data: LSTM scores for all tweets
lstm_scores = {}
with open('../../../datasets/HateSPic/twitter/'+scores_file+'.txt') as f:
    for line in f:
        data = line.split(',')
        lstm_scores[int(data[0])] = float(data[1])

# Load test indices
results = []
with open('../../../datasets/HateSPic/HateSPic/tweet_embeddings/MMHS-v3mm-lstm_embeddings_test_hate.txt') as f:
    for line in f:
        data = line.split(',')
        try:
            score = (lstm_scores[int(data[0])] + visual_model_data[int(data[0])]) / 2
            results.append([1, score, int(data[0])])
        except:
            print("LSTM score for " + str(data[0]) + " not found, continuing")
            continue

with open('../../../datasets/HateSPic/HateSPic/tweet_embeddings/MMHS-v3mm-lstm_embeddings_test_nothate.txt') as f:
    for line in f:
        data = line.split(',')
        try:
            score = (lstm_scores[int(data[0])] + visual_model_data[int(data[0])]) / 2
            results.append([0, score, int(data[0])])
        except:
            print("LSTM score for " + str(data[0]) + " not found, continuing")
            continue

model_name = "Ensemble"
random.shuffle(results)
evaluate(results, model_name)

