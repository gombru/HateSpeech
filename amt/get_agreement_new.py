import json

gt = json.load(open("../../../datasets/HateSPic/MMHS/anns/analysis/50k_3workers.json"))

out_ids_agg_nothate = open("../../../datasets/HateSPic/MMHS/anns/analysis/ids_agreement_nothate.txt","w")
out_ids_agg_hate = open("../../../datasets/HateSPic/MMHS/anns/analysis/ids_agreement_hate.txt","w")
out_ids_diss = open("../../../datasets/HateSPic/MMHS/anns/analysis/ids_disagreement.txt","w")

out_ids_agg_sexist = open("../../../datasets/HateSPic/MMHS/anns/analysis/ids_agreement_sexist.txt","w")
out_ids_agg_homo = open("../../../datasets/HateSPic/MMHS/anns/analysis/ids_agreement_homo.txt","w")
out_ids_agg_racist = open("../../../datasets/HateSPic/MMHS/anns/analysis/ids_agreement_racist.txt","w")


agreement = {'hate':0, 'NotHate':0}
total = {'hate':0, 'NotHate':0}
agreement_classes = {"NotHate": 0, "Racist": 0, "Sexist": 0, "Homophobe": 0, "Religion": 0, "OtherHate": 0}
total_classes = {"NotHate": 0, "Racist": 0, "Sexist": 0, "Homophobe": 0, "Religion": 0, "OtherHate": 0}
ids_agreement_nothate = []
ids_agreement_hate = []
ids_disagreement = []

ids_agreement_sexist = []
ids_agreement_homo = []
ids_agreement_racist = []

count=0

for k,v in gt.items():

    v1 = v['labels'][0]
    v2 = v['labels'][1]
    v3 = v['labels'][2]

    # if v['label'] == 5: continue
    count+=1
    if v1 == 0:
        total['NotHate']+=1
        total_classes['NotHate']+=1
        if v2 == 0 and v3 == 0:
            agreement['NotHate']+=1
            agreement_classes['NotHate']+=1
            ids_agreement_nothate.append(k)
        else:
            ids_disagreement.append(k)
    else:
        total['hate'] += 1
        total_classes[v['labels_str'][0]]+=1
        if v2 > 0 and v3 > 0:
            agreement['hate']+=1
            ids_agreement_hate.append(k)
        else:
            ids_disagreement.append(k)
        if v2 == v1 and v3 == v1:
            agreement_classes[v['labels_str'][0]]+=1
            if "exist" in v['labels_str'][0]:
                ids_agreement_sexist.append(k)
            if "omo" in v['labels_str'][0]:
                ids_agreement_homo.append(k)
            if "acist" in v['labels_str'][0]:
                ids_agreement_racist.append(k)


for el in ids_disagreement:
    out_ids_diss.write(str(el) + '\n')
for el in ids_agreement_nothate:
    out_ids_agg_nothate.write(str(el) + '\n')
for el in ids_agreement_hate:
    out_ids_agg_hate.write(str(el) + '\n')

for el in ids_agreement_sexist:
    out_ids_agg_sexist.write(str(el) + '\n')
for el in ids_agreement_homo:
    out_ids_agg_homo.write(str(el) + '\n')
for el in ids_agreement_racist:
    out_ids_agg_racist.write(str(el) + '\n')

print("Total annotator 1")
print(total)
print("Agreement")
print(agreement)
print("------------------")
print("Total per class annotator 1")
print(total_classes)
print("Agreement per class")
print(agreement_classes)
print("------------------")
print("Total agreement:")
print(str(count - len(ids_disagreement)) + "/" + str(count))
print(str(100 * (float((count - len(ids_disagreement))) / count)))


