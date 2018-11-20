from evaluate import evaluate
import random


model_name = 'LSTM_niggaFaggot_tweet_text'
scores_file = 'MMHS_niggaFaggot_lstm_scores' #'lstm_scores'

# Load data: LSTM scores for all tweets
lstm_scores = {}
with open('../../../datasets/HateSPic/twitter/'+scores_file+'.txt') as f:
    for line in f:
        data = line.split(',')
        lstm_scores[int(data[0])] = float(data[1])

# Load test indices
results = []
with open('../../../datasets/HateSPic/HateSPic/tweet_embeddings/MMHS-niggaFaggot-lstm_embeddings_test_hate.txt') as f:
    for line in f:
        data = line.split(',')
        try:
            results.append([1, lstm_scores[int(data[0])], int(data[0])])
        except:
            print("LSTM score for " + str(data[0]) + " not found, continuing")
            continue

with open('../../../datasets/HateSPic/HateSPic/tweet_embeddings/MMHS-niggaFaggot-lstm_embeddings_test_nothate.txt') as f:
    for line in f:
        data = line.split(',')
        try:
            results.append([0, lstm_scores[int(data[0])], int(data[0])])
        except:
            print("LSTM score for " + str(data[0]) + " not found, continuing")
            continue

random.shuffle(results)
evaluate(results, model_name)