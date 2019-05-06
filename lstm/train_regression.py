# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
# import classification_datasets
# import MMHS50K_dataset_MMLSTM
import MMHS_dataset_regression
# import SynthMMHS_dataset
import os
import random
torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)
torch.cuda.set_device(0)
import torch.utils.data as Data

base_path = "../../../datasets/HateSPic/MMHS/lstm_models/"

class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, batch_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, 1) # 1 is label size (hate value)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())) # 2 if bidirectional

        # return autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        #         # No cell state for GRU

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

def train():
    id = 'MMHS_regression_hidden_150_embedding_100_best_model'
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 150 #50 #150
    EPOCH = 10
    BATCH_SIZE = 10 #10
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    train_iter, dev_iter = MMHS_dataset_regression.load_MMHS50K(text_field, label_field, batch_size=BATCH_SIZE)
    print("Len labels vocab: " + str(len(label_field.vocab)))
    print(label_field.vocab.itos)
    print("Used label len is: " + str(len(label_field.vocab)-1))

    text_field.vocab.load_vectors('glove.twitter.27B.100d') #mmhsv3_glove.twitter.27B.200d

    model = LSTMClassifier(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                           vocab_size=len(text_field.vocab),
                            batch_size=BATCH_SIZE)
    model = model.cuda()
    model.word_embeddings.weight.data = text_field.vocab.vectors.cuda()
    loss_function = nn.MSELoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3) #1e-3
    model = model.cuda()

    no_up = 0
    best_dev_loss = 1000
    for i in range(EPOCH):
        print('epoch: %d start!' % i)
        train_epoch(model, train_iter, loss_function, optimizer, text_field, label_field, i)
        dev_loss = evaluate(model,dev_iter,loss_function,'dev')
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            os.system('rm ' + base_path + id + '*.model')
            print('New Best Dev Loss!!! '  + str(best_dev_loss))
            torch.save(model.state_dict(), base_path + id + '.model')
            no_up = 0
        else:
            print('NOT improving Best Dev Loss')
            no_up += 1
            if no_up >= 10:
                print('Ending because the DEV ACC does not improve')
                exit()
#
def evaluate(model, eval_iter, loss_function,  name ='dev'):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    count = 0
    for batch in eval_iter:
        sent, label = batch.text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        regression_labels = torch.zeros([model.batch_size,1], dtype=torch.float32)
        for c in range(0,model.batch_size):
            regression_labels[c] = batch.dataset.examples[count+c].label
        model.hidden = model.init_hidden()  # detaching it from its history on the last instance.
        pred = model(sent.cuda())
        loss = loss_function(pred.cpu(), regression_labels)
        avg_loss += loss.data[0]
    avg_loss /= len(eval_iter)
    print(name + ' avg_loss:%g ' % (avg_loss))
    return avg_loss


def train_epoch(model, train_iter, loss_function, optimizer, text_field, label_field, i):
    model.train()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    count = 0
    for batch in train_iter:
        sent, label = batch.text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        regression_labels = torch.zeros([model.batch_size,1], dtype=torch.float32)
        for c in range(0,model.batch_size):
            regression_labels[c] = batch.dataset.examples[count+c].label
        model.hidden = model.init_hidden()# detaching it from its history on the last instance.
        pred = model(sent.cuda())
        pred_label = pred.cpu().data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        model.zero_grad()
        loss = loss_function(pred, regression_labels.cuda())
        avg_loss += loss.data[0]
        count += model.batch_size
        if count % 1000 == 0:
            print('epoch: %d iterations: %d loss :%g' % (i, count*model.batch_size, loss.data[0]))
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_iter)
    print('epoch: %d done!\ntrain avg_loss:%g , acc:%g'%(i, avg_loss, get_accuracy(truth_res,pred_res)))

train()
