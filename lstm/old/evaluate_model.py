# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
# import classification_datasets
import MMHS50K_dataset_test
import os
import random
torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)
#torch.cuda.set_device(0)
import torch.utils.data as Data

base_path = "../../../datasets/HateSPic/MMHS50K/lstm_models/"

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
        return log_probs

def get_accuracy(truth, pred):
     assert len(truth)==len(pred)
     right = 0
     for i in range(len(truth)):
         if truth[i]==pred[i]:
             right += 1.0
     return right/len(truth)

def test():
    model_path = '../../../datasets/HateSPic/MMHS50K/lstm_models/MMHS50K_niggaFaggot_hidden_150_embedding_100_best_model_acc_val57.model'
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 150 #50
    BATCH_SIZE = 10
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    train_iter, dev_iter = MMHS50K_dataset_test.load_MMHS50K(text_field, label_field, batch_size=BATCH_SIZE)

    text_field.vocab.load_vectors('glove.twitter.27B.100d')

    model = LSTMClassifier(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                           vocab_size=len(text_field.vocab),label_size=len(label_field.vocab)-1,
                            batch_size=BATCH_SIZE)

    model.word_embeddings.weight.data = text_field.vocab.vectors
    model.load_state_dict((torch.load(model_path)))
    evaluate(model,dev_iter)

#
def evaluate(model, eval_iter):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    for batch in eval_iter:
        sent, label = batch.text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()  # detaching it from its history on the last instance.
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        # pred_res += [x[0] for x in pred_label]
        pred_res += [x for x in pred_label]

    avg_loss /= len(eval_iter)
    acc = get_accuracy(truth_res, pred_res)
    print('val acc:%g' % (avg_loss, acc ))

test()
