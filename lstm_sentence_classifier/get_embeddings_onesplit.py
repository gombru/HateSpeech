# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
# import classification_datasets
import hate_dataset_test
import os
import random
import numpy as np
torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)
# torch.cuda.set_device(0)
import torch.utils.data as Data

out_file_name = 'test'
split = 'test'
out_file = open("../../datasets/HateSPic/lstm_emeddings/" + out_file_name + ".txt",'w')
out_file_classification = open("../../datasets/HateSPic/lstm_embeddings/" + out_file_name + "_classification.txt",'w')

classes =['hate','nonhate']

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
        return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size , -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
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
    model_path = './best_models/hate_annotated_best_model_minibatch_acc_63.model'
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 50
    BATCH_SIZE = 1
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    id_field = data.Field()
    split_iter = hate_dataset_test.load_HD(text_field, label_field, id_field, batch_size=BATCH_SIZE, split = split)

    text_field.vocab.load_vectors('glove.twitter.27B.100d')

    model = LSTMClassifier(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                           vocab_size=len(text_field.vocab),label_size=len(label_field.vocab)-1,
                            batch_size=BATCH_SIZE)
    model.word_embeddings.weight.data = text_field.vocab.vectors
    model.load_state_dict((torch.load(model_path)))

    print len(split_iter)

    print "Computing ..."
    evaluate(model,split_iter)




def evaluate(model, split_iter):
    model.eval()
    count = 0
    results = []
    results_classification = []
    correct = 0
    for it in split_iter:
        if count % 1000 == 0 : print count
        sent, label = it.text, it.label
        label_text = it.dataset.examples[count].label
        # print label_text + "  " + str(label.data[0])
        id = it.dataset.examples[count].id[0]
        label.data.sub_(1)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()  # detaching it from its history on the last instance.
        pred, hidden = model(sent)
        embedding = np.zeros(50)
        for c,d in enumerate(hidden[0][0,0,:]):
            embedding[c] = d.data[0]

        if min(embedding) < 0:
            embedding = embedding - min(embedding)
        if sum(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)

        embeddingString = ""
        for d in embedding:
            embeddingString += ',' + str(d)

        gt_label = classes.index(label_text)
        predicted_label = pred.data.max(1)[1].numpy()
        cur_correct = 0
        if label.data[0] == predicted_label:
            correct+=1
            cur_correct = 1

        results_classification += id + ',' + str(cur_correct) + '\n'

        results += id + "," + str(gt_label), embeddingString + '\n'
        count += 1

    print "Correct labels: " + str(correct) + " / " + str(count) + " --> " + str(float(correct)/count*100)

    print "Writing results"
    for i,r in enumerate(results):
        out_file.write(r)


    for i,r in enumerate(results_classification):
        out_file_classification.write(r)


test()
out_file.close()
out_file_classification.close()

print "DONE"