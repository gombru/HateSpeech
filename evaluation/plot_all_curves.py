import os
import numpy as np
import matplotlib.pyplot as plt
import json


data = {}

colors = ['r','g','b','m','k','c','#f47d42','#f4ca41','#c1f441','#7641f4']

for file in os.listdir('../../../datasets/HateSPic/HateSPic/evaluation_results/'):
    with open('../../../datasets/HateSPic/HateSPic/evaluation_results/' + file) as f:
        data[file] = json.load(f)

legends = []
for k, v in data.iteritems():
    if 'onlyTweet' in k: legends.append('FCM - TT')
    elif 'onlyText' in k: legends.append('FCM - TT,IT')
    elif 'onlyI' in k: legends.append('FCM - I')
    elif 'Random' in k: legends.append('Random')
    elif 'LSTM' in k: legends.append('LSTM - TT')
    elif 'Concat_SameSameDim' in k: legends.append('SCM - All')
    elif 'TextualKernels' in k: legends.append('TKM - All')
    else: legends.append('FCM - All')


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
