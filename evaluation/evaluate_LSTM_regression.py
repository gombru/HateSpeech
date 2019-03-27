from evaluate import evaluate
import random


model_name = 'MMHS_regression_hidden_150_embedding_100_best_model'
scores_file = 'MMHS_regression_hidden_150_embedding_100_best_model' #'lstm_scores'

# Load data: LSTM scores for all tweets
lstm_scores = {}
with open('../../../datasets/HateSPic/MMHS/lstm_scores/'+scores_file+'.txt') as f:
    for line in f:
        data = line.split(',')
        lstm_scores[int(data[0])] = float(data[1])

# Load test indices
results = []
with open('../../../datasets/HateSPic/MMHS/lstm_data/lstm_data_50k_3workers_regression/tweets.test') as f:
    for line in f:
        data = line.split(',')
        try:
            label = 0
            if float(data[2]) > 0.5: label = 1
            results.append([label, lstm_scores[int(data[0])], int(data[0])])
        except:
            print("LSTM score for " + str(data[0]) + " not found, continuing")
            continue


random.shuffle(results)
evaluate(results, model_name)