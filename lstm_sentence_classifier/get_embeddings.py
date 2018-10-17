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

target = 'train_nothate'
split_name = 'tweets.' + target
model_name = 'MMHS10K-v2mm_dataset_hidden_50_best_model_minibatch_acc_80' # 'saved_hate_annotated_hidden_50_best_model_minibatch_acc_77'
out_file_name = 'tweet_embeddings/MMHS-v2mm-lstm_embeddings_' + target
out_file = open("../../../datasets/HateSPic/HateSPic/" + out_file_name + ".txt",'w')
split_folder = 'HateSPic_v2mm'

classes =['hate','nothate']

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
    model_path = '../../../datasets/HateSPic/lstm_models/' + model_name + '.model'
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 50
    BATCH_SIZE = 1
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    id_field = data.Field(sequential=False, use_vocab=False)
    split_iter = hate_dataset_test.load_HD(text_field, label_field, id_field, batch_size=BATCH_SIZE, split_folder=split_folder, split_name = split_name)

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
    results_string = ''

    for it in split_iter:
        if count % 100 == 0: print count
        print count
        sent, label = it.text, it.label
        text = it.dataset.examples[count].text
        text_str = ''
        for w in text:
            try:
                text_str += w.decode('utf-8') + ' '
            except:
                continue
        label_text = it.dataset.examples[count].label
        id = it.dataset.examples[count].id
        model.batch_size = 1
        model.hidden = model.init_hidden()  # detaching it from its history on the last instance.
        pred, hidden = model(sent)


        # Get embedding stre
        embedding = np.zeros(50)
        for c,d in enumerate(hidden[0][0,0,:]):
            embedding[c] = d.data[0]
        embeddingString = ""
        for d in embedding:
            embeddingString += ',' + str(d)

        results_string += id + embeddingString + '\n'
        count += 1

    print "Writing results"
    out_file.write(results_string)

test()
out_file.close()

print "DONE"