# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
# import classification_datasets
import MMHS50K_dataset_test
# import SynthMMHS_dataset_test
# import MMHS50K_dataset_classes_test
import os
import random
import numpy as np
torch.set_num_threads(4)
torch.manual_seed(1)
random.seed(1)
torch.cuda.set_device(0)

target = 'img_txt'
split_name = 'tweets.' + target
# split_name = 'anns_val_hate.txt'
# target = 'val_hate'
model_name = 'MMHS_classification_hidden_150_embedding_100_best_model_acc_val61' # 'saved_hate_annotated_hidden_50_best_model_minibatch_acc_77'
out_file_name = 'tweet_embeddings/MMHS_lstm_embeddings_classification/' + target
out_file = open("../../../datasets/HateSPic/MMHS/" + out_file_name + ".txt",'w')
split_folder = ''

class_labels =['hate','nothate']
# class_labels = ['NotHate','Racist','Sexist','Homophobe']

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
    model_path = '../../../datasets/HateSPic/MMHS/lstm_models/' + model_name + '.model'
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 150
    BATCH_SIZE = 8 # 2048, 128
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    id_field = data.Field(sequential=False, use_vocab=False)
    split_iter = MMHS50K_dataset_test.load_MMHS50K(text_field, label_field, id_field, batch_size=BATCH_SIZE, split_folder=split_folder, split_name = split_name)
    print("Len labels vocab: " + str(len(label_field.vocab)))

    text_field.vocab.load_vectors('glove.twitter.27B.100d')

    model = LSTMClassifier(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                           vocab_size=len(text_field.vocab),label_size=len(class_labels),
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

        # if count % 100 == 0: print count
        print(count)

        sent, label = batch.text, batch.label
        cur_batch_size = label.__len__()
        model.batch_size = cur_batch_size
        model.hidden = model.init_hidden()  # detaching it from its history on the last instance.
        pred, hidden = model(sent.cuda())
        pred_label = pred.cpu().data.max(1)[1].numpy()

        # Get embedding per batch element
        for i in range(0, cur_batch_size):
            el_embedding = hidden[0][0,i,:]
            id = batch.dataset.examples[count + i].id
            embeddingString = ""
            for d in el_embedding: embeddingString += ',' + str(d.item())
            result_string = id + embeddingString + '\n'
            out_file.write(result_string)
            predicted_classes_count[pred_label[i]] += 1
        count+= cur_batch_size

    print("Writing results")
    out_file.close()

    print("Predicted classes:")
    for i,cl in enumerate(class_labels):
        print(cl + ": " + str(predicted_classes_count[i]))

test()
print("DONE")