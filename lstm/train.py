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
import MMHS50K_dataset
# import SynthMMHS_dataset
import os
import random
torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)
torch.cuda.set_device(0)
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
        return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())) # 2 if bidirectional

        # return autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        #         # No cell state for GRU

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

def train():
    id = 'MMHS50K_niggaNigger_hidden_150_embedding_100_best_model_acc_val'
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 150 #50 #150
    EPOCH = 10
    BATCH_SIZE = 10 #10
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    train_iter, dev_iter = MMHS50K_dataset.load_MMHS50K(text_field, label_field, batch_size=BATCH_SIZE)
    print("Len labels vocab: " + str(len(label_field.vocab)))
    print(label_field.vocab.itos)
    print("Used label len is: " + str(len(label_field.vocab)-1))


    # class_labels = ['NotHate', 'Racist', 'Sexist', 'Homophobe']
    # class_weights = [18883, 4833, 2414, 2400] # Weights for classes

    class_labels = ['NotHate', 'Hate']
    # class_weights = [9502,17535] # Weights for noOtherHard
    class_weights = [1,1]

    min_instances = min(class_weights)
    for i in range(0,len(class_weights)):
        class_weights[i] = 1 / (float(class_weights[i]) / min_instances)
    class_weights = torch.FloatTensor(class_weights).cuda()
    print("Class weights: ")
    print(class_weights)
    text_field.vocab.load_vectors('glove.twitter.27B.100d') #mmhsv3_glove.twitter.27B.200d
    #text_field.vocab.load_vectors(wv_type='glove.6B', wv_dim=100)

    best_dev_acc = 0.0

    model = LSTMClassifier(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                           vocab_size=len(text_field.vocab),label_size=len(label_field.vocab)-1,
                            batch_size=BATCH_SIZE)
    model = model.cuda()
    model.word_embeddings.weight.data = text_field.vocab.vectors.cuda()
    loss_function = nn.NLLLoss(weight=class_weights).cuda()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3) #1e-3
    model = model.cuda()

    no_up = 0
    for i in range(EPOCH):
        print('epoch: %d start!' % i)
        train_epoch(model, train_iter, loss_function, optimizer, text_field, label_field, i)
        print(id + ' --> now best dev acc:',best_dev_acc)
        dev_acc = evaluate_classes(class_labels, model,dev_iter,loss_function,'dev')
        # dev_acc = evaluate(model,dev_iter,loss_function,'dev')
        # test_acc = evaluate(model, test_iter, loss_function,'test')
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            os.system('rm ' + base_path + id + '*.model')
            print('New Best Dev!!! '  + str(best_dev_acc))
            torch.save(model.state_dict(), base_path + id + str(int(dev_acc*100)) + '.model')
            no_up = 0
        else:
            print('NOT improving Best Dev')
            no_up += 1
            if no_up >= 10:
                print('Ending because the DEV ACC does not improve')
                exit()
#
def evaluate(model, eval_iter, loss_function,  name ='dev'):
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
        pred = model(sent.cuda())
        pred_label = pred.cpu().data.max(1)[1].numpy()
        # pred_res += [x[0] for x in pred_label]
        pred_res += [x for x in pred_label]
        loss = loss_function(pred.cpu(), label)
        avg_loss += loss.data[0]

    avg_loss /= len(eval_iter)
    acc = get_accuracy(truth_res, pred_res)
    print(name + ' avg_loss:%g  acc:%g' % (avg_loss, acc ))
    return acc

def evaluate_classes(class_labels, model, eval_iter, loss_function,  name ='dev'):
    model.eval()
    truth_res = []
    pred_res = []

    tp_classes = np.zeros(len(class_labels))
    fn_classes = np.zeros(len(class_labels))
    accuracies = np.zeros(len(class_labels))

    for batch in eval_iter:
        sent, label = batch.text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()  # detaching it from its history on the last instance.
        pred = model(sent.cuda())
        pred_label = pred.cpu().data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]

    for i,cur_pred_res in enumerate(pred_res):
        if cur_pred_res == truth_res[i]: tp_classes[truth_res[i]] += 1
        else: fn_classes[truth_res[i]] += 1

    for i in range(0,len(class_labels)):
        accuracies[i] = tp_classes[i] / float((tp_classes[i] + fn_classes[i]))

    acc = accuracies.mean()

    print(name + ' acc:%g' % (acc ))
    return acc

def train_epoch(model, train_iter, loss_function, optimizer, text_field, label_field, i):
    model.train()
    avg_loss = 0.0
    count = 0
    truth_res = []
    pred_res = []
    for batch in train_iter:
        sent, label = batch.text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()# detaching it from its history on the last instance.
        pred = model(sent.cuda())
        pred_label = pred.cpu().data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        model.zero_grad()
        loss = loss_function(pred, label.cuda())
        avg_loss += loss.data[0]
        count += 1
        if count % 1000 == 0:
            print('epoch: %d iterations: %d loss :%g' % (i, count*model.batch_size, loss.data[0]))
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_iter)
    print('epoch: %d done!\ntrain avg_loss:%g , acc:%g'%(i, avg_loss, get_accuracy(truth_res,pred_res)))

train()
