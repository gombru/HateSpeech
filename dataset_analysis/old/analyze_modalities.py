import json
import numpy as np

data_path = '../../../datasets/HateSPic/MMHS/anns/1k_3workers_v2.json'
print("Loading data ...")
data = json.load(open(data_path,'r'))

agreement_labels = 0
agreement_modalities = 0

ids_image = []
ids_both = []
ids_combined = []

ids_hate = []
ids_not_hate = []

ids_racist = []
ids_sexist = []
ids_homo = []
ids_religion = []


c=0

agreement_notHate = 0
dissagreement_notHate = 0

for k,v in data.iteritems():
    c+=1
    labels = v['labels']
    mods = v['modalities']

    if labels[0] == labels[1] or labels[0] == labels[2] or labels[1] == labels[2]:
        label = np.bincount(labels).argmax()
        if label == 1: ids_racist.append(k)
        if label == 2: ids_sexist.append(k)
        if label == 3: ids_homo.append(k)
        if label == 4: ids_religion.append(k)

    for i in range(0,3):
        if labels[i] > 0: labels[i] = 1

    if labels[0] == labels[1] and labels[0] == labels[2]:
        agreement_labels+=1
        if mods[0] == mods[1] and mods[0] == mods[2]:
            agreement_modalities += 1
        if labels[0] == 0: ids_hate.append(k)
        else: ids_not_hate.append(k)





    if mods[0] == mods[1] or mods[0] == mods[2] or mods[1] == mods[2]:
        mod = np.bincount(mods).argmax()
        if mod == 2: ids_image.append(k)
        if mod == 3: ids_both.append(k)
        if mod == 4: ids_combined.append(k)



with open('../../../datasets/HateSPic/MMHS/anns/analysis/modalities_ids_image.txt','w') as out_file:
    for id in ids_image: out_file.write(str(id) + '\n')
with open('../../../datasets/HateSPic/MMHS/anns/analysis/modalities_ids_both.txt', 'w') as out_file:
    for id in ids_both: out_file.write(str(id) + '\n')
with open('../../../datasets/HateSPic/MMHS/anns/analysis/modalities_ids_combined.txt', 'w') as out_file:
    for id in ids_combined: out_file.write(str(id) + '\n')

with open('../../../datasets/HateSPic/MMHS/anns/analysis/ids_hate.txt', 'w') as out_file:
    for id in ids_hate: out_file.write(str(id) + '\n')
with open('../../../datasets/HateSPic/MMHS/anns/analysis/ids_notHate.txt', 'w') as out_file:
    for id in ids_not_hate: out_file.write(str(id) + '\n')

with open('../../../datasets/HateSPic/MMHS/anns/analysis/ids_racist.txt', 'w') as out_file:
    for id in ids_racist: out_file.write(str(id) + '\n')
with open('../../../datasets/HateSPic/MMHS/anns/analysis/ids_sexist.txt', 'w') as out_file:
    for id in ids_sexist: out_file.write(str(id) + '\n')
with open('../../../datasets/HateSPic/MMHS/anns/analysis/ids_homo.txt', 'w') as out_file:
    for id in ids_homo: out_file.write(str(id) + '\n')
with open('../../../datasets/HateSPic/MMHS/anns/analysis/ids_religion.txt', 'w') as out_file:
    for id in ids_religion: out_file.write(str(id) + '\n')

print(c)
print(agreement_labels)
print(agreement_modalities)

print(ids_image)
print(ids_both)
print(ids_combined)


print(agreement_notHate)
print(dissagreement_notHate)