from evaluate import evaluate
import random


model_name = 'MMHS_classification_hidden_150_embedding_100_best_model_acc_val61'
scores_file = 'MMHS_classification_hidden_150_embedding_100_best_model_acc_val61' #'lstm_scores'

# Load data: LSTM scores for all tweets
lstm_scores = {}
with open('../../../datasets/HateSPic/MMHS/lstm_scores/'+scores_file+'.txt') as f:
    for line in f:
        data = line.split(',')
        lstm_scores[int(data[0])] = float(data[1])

# Load test indices
results = []
with open('../../../datasets/HateSPic/MMHS/tweet_embeddings/MMHS_lstm_embeddings_classification/test_hate.txt') as f:
    for line in f:
        data = line.split(',')
        try:
            results.append([1, lstm_scores[int(data[0])], int(data[0])])
        except:
            print("LSTM score for " + str(data[0]) + " not found, continuing")
            continue

with open('../../../datasets/HateSPic/MMHS/tweet_embeddings/MMHS_lstm_embeddings_classification/test_nothate.txt') as f:
    for line in f:
        data = line.split(',')
        try:
            results.append([0, lstm_scores[int(data[0])], int(data[0])])
        except:
            print("LSTM score for " + str(data[0]) + " not found, continuing")
            continue

random.shuffle(results)
evaluate(results, model_name)