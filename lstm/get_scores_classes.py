# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
import MMHS50K_dataset_classes_test
import os
import random
import numpy as np
torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)
torch.cuda.set_device(0)

target = 'test'
out_file_name = 'MMHS50K_classes_lstm_scores'
split_name = 'tweets.' + target
out_file = open("../../../datasets/HateSPic/MMHS50K/lstm_scores/" + out_file_name + ".txt",'w')
split_folder = ''
model_name = 'MMHS50K_classes_hidden_50_embedding_100_best_model_acc_val72'

class_labels = ['NotHate','Racist','Sexist','Homophobe']

class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
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
        log_probs = F.log_softmax(y)
        return log_probs, self.hidden

def get_accuracy(truth, pred):
     assert len(truth)==len(pred)
     right = 0
     for i in range(len(truth)):
         if truth[i]==pred[i]:
             right += 1.0
     return right/len(truth)

def test():
    model_path = '../../../datasets/HateSPic/MMHS50K/lstm_models/' + model_name + '.model'
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 150
    BATCH_SIZE = 2048
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    id_field = data.Field(sequential=False, use_vocab=False)
    split_iter = MMHS50K_dataset_classes_test.load_MMHS50K(text_field, label_field, id_field, batch_size=BATCH_SIZE, split_folder=split_folder, split_name = split_name)

    text_field.vocab.load_vectors('glove.twitter.27B.100d')

    model = LSTMClassifier(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                           vocab_size=len(text_field.vocab),label_size=len(label_field.vocab)-1,
                            batch_size=BATCH_SIZE)
    model.word_embeddings.weight.data = text_field.vocab.vectors
    model.load_state_dict((torch.load(model_path)))
    model = model.cuda()
    print len(split_iter)

    print "Computing ..."
    evaluate(model,split_iter)




def evaluate(model, split_iter):

    predicted_classes_count = np.zeros(len(class_labels))
    model.eval()
    count = 0
    results_string = ''

    for batch in split_iter:

        print count

        sent, label = batch.text, batch.label

        cur_batch_size = label.__len__()
        model.batch_size = cur_batch_size
        model.hidden = model.init_hidden()  # detaching it from its history on the last instance.
        pred, hidden = model(sent.cuda())
        pred_label = pred.cpu().data.max(1)[1].numpy()
        # Get score per batch element
        for i in range(0, cur_batch_size):

            # Get text
            text = batch.dataset.examples[count+i].text
            text_str = ''
            for w in text:
                try:
                    text_str += w.decode('utf-8') + ' '
                except:
                    continue

            # Get class scores
            class_scores = ''
            for class_score in pred[i]:
                class_scores += str(float(class_score)) + ','
            id = batch.dataset.examples[count + i].id
            label_str = batch.dataset.examples[count + i].label
            label_id = ["NotHate", "Racist", "Sexist", "Homophobe"].index(str(label_str))
            results_string += id + ',' + str(label_id) + ',' + class_scores + text_str + '\n'
            predicted_classes_count[pred_label[i]] += 1

        count += cur_batch_size

    print "Writing results"
    out_file.write(results_string)
    out_file.close()

    print("Predicted classes:")
    for i,cl in enumerate(class_labels):
        print(cl + ": " + str(predicted_classes_count[i]))

test()
print "DONE"