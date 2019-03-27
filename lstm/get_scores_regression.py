# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
# import classification_datasets
import MMHS_dataset_regression_test
import os
import random
import numpy as np
torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)
torch.cuda.set_device(0)

target = 'test'
out_file_name = 'MMHS_regression_hidden_150_embedding_100_best_model'
split_name = 'tweets.' + target
out_file = open("../../../datasets/HateSPic/MMHS/lstm_scores/" + out_file_name + ".txt",'w')
split_folder = ''
model_name = 'MMHS_regression_hidden_150_embedding_100_best_model' # 'saved_hate_annotated_hidden_50_best_model_minibatch_acc_77'

class_labels =['hate','nothate']

class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, 1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size , -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.hidden2label(lstm_out[-1])
        return y

def get_accuracy(truth, pred):
     assert len(truth)==len(pred)
     right = 0
     for i in range(len(truth)):
         if truth[i]==pred[i]:
             right += 1.0
     return right/len(truth)

def test():
    model_path = '../../../datasets/HateSPic/MMHS/lstm_models/' + model_name + '.model'
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 150
    BATCH_SIZE = 64 # 2048
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    id_field = data.Field(sequential=False, use_vocab=False)
    split_iter = MMHS_dataset_regression_test.load_MMHS50K(text_field, label_field, id_field, batch_size=BATCH_SIZE, split_folder=split_folder, split_name = split_name)

    text_field.vocab.load_vectors('glove.twitter.27B.100d')

    model = LSTMClassifier(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                           vocab_size=len(text_field.vocab),label_size=len(label_field.vocab)-1,
                            batch_size=BATCH_SIZE)
    model.word_embeddings.weight.data = text_field.vocab.vectors
    model.load_state_dict((torch.load(model_path)))
    model = model.cuda()
    print(len(split_iter))

    print("Computing ...")
    evaluate(model,split_iter)




def evaluate(model, split_iter):

    predicted_classes_count = np.zeros(len(class_labels))
    model.eval()
    count = 0
    results_string = ''

    for batch in split_iter:

        print(count)

        sent, label = batch.text, batch.label

        cur_batch_size = label.__len__()
        model.batch_size = cur_batch_size

        regression_labels = torch.zeros([model.batch_size,1], dtype=torch.float32)
        for c in range(0,model.batch_size):
            regression_labels[c] = batch.dataset.examples[count+c].label

        model.hidden = model.init_hidden()  # detaching it from its history on the last instance.
        pred = model(sent.cuda())
        pred = pred.cpu().data.numpy()

        # Get score per batch element
        for i in range(0, cur_batch_size):
            text = batch.dataset.examples[count+i].text
            text_str = ''
            for w in text:
                try:
                    text_str += w.decode('utf-8') + ' '
                except:
                    continue
            id = batch.dataset.examples[count+i].id
            results_string += id + ',' + str(pred[i][0]) + ',' + text_str + '\n'

        count += cur_batch_size

    print("Writing results")
    out_file.write(results_string)
    out_file.close()

    print("Predicted classes:")
    for i,cl in enumerate(class_labels):
        print(cl + ": " + str(predicted_classes_count[i]))

test()
print("DONE")