from evaluate_classes import evaluate_classes
import random

model_name = 'LSTM_tweet_text'
scores_file = 'MMHS50K_classes_lstm_scores' #'lstm_scores'

# Load data: LSTM scores, label and indices for test tweets
results = [] # [class_label, NotHate_score, Racist_score, Sexism_score, Homophobe_score, id]
with open('../../../datasets/HateSPic/MMHS50K/lstm_scores/'+scores_file+'.txt') as f:
    for line in f:
        d = line.split(',')
        results.append([int(d[1]), float(d[2]), float(d[3]), float(d[4]), float(d[5]), int(d[0])])
evaluate_classes(results)