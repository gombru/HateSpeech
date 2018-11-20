from evaluate import evaluate
import numpy as np
import random


model_name = 'Davison'

# Load test Labels
hate_ids = []
with open('../../../datasets/HateSPic/HateSPic/tweet_embeddings/MMHS-v2mm-lstm_embeddings_test_hate.txt') as f:
    for line in f:
        data = line.split(',')
        try:
            hate_ids.append(data[0])
        except:
            print("LSTM score for " + str(data[0]) + " not found, continuing")
            continue

results = []
with open('../../../datasets/HateSPic/HateSPic/davison/MMHS10K_v2mm_testScores.txt') as f:
    for line in f:
        data = line.split(',')
        label = 0
        if data[0] in hate_ids: label = 1
        hate_score = float(data[2])
        notHate_score = float(data[3])
        softmax_hate_score = np.exp(hate_score) / (np.exp(hate_score) + np.exp(notHate_score))
        results.append([label, softmax_hate_score])
evaluate(results, model_name)