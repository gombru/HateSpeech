import os
import numpy as np
import matplotlib.pyplot as plt
import json


data = {}

colors = ['r','g','b','m','k','c']

for file in os.listdir('../../../datasets/HateSPic/HateSPic/evaluation_results/'):
    with open('../../../datasets/HateSPic/HateSPic/evaluation_results/' + file) as f:
        data[file] = json.load(f)

legends = []
for k, v in data.iteritems():
    if 'onlyTweet' in k: legends.append('Only Tweet Text')
    elif 'onlyText' in k: legends.append('Only Text')
    elif 'onlyImage' in k: legends.append('Only Image')
    elif 'Random' in k: legends.append('Random')
    elif 'LSTM' in k: legends.append('LSTM Only Tweet Text')
    else: legends.append('Full Tweet')


c=0
for k, v in data.iteritems():
    plt.plot(v['recalls'], v['precisions'], colors[c], label=legends[c])
    c+=1

plt.ylabel('Precision')
plt.xlabel('Recall')
plt.legend()
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.show()

c=0
for k, v in data.iteritems():
    plt.plot(v['recalls'], v['fpr'], colors[c], label=legends[c])
    c+=1
# Print ROC curve
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.show()
