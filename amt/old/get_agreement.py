import json

worker_one_GT = json.load(open("../../../datasets/HateSPic/MMHS/anns/MMHS_5K_GT_worker2.json"))
worker_two_GT = json.load(open("../../../datasets/HateSPic/MMHS/anns/MMHS_5K_GT_worker3.json"))
out_file = open("../../../datasets/HateSPic/AMT/disagreement/disagreement_5k.txt",'w')

agreement = {'hate':0, 'NotHate':0}
total = {'hate':0, 'NotHate':0}
agreement_classes = {"NotHate": 0, "Racist": 0, "Sexist": 0, "Homophobe": 0, "Religion": 0, "OtherHate": 0}
total_classes = {"NotHate": 0, "Racist": 0, "Sexist": 0, "Homophobe": 0, "Religion": 0, "OtherHate": 0}
ids_disagreed = []
count=0
for k,v in worker_one_GT.iteritems():
    try:
        v2 = worker_two_GT[k]
        # if v['label'] == 5: continue
        count+=1
        if v['label'] == 0:
            total['NotHate']+=1
            total_classes['NotHate']+=1
            if v2['label'] == 0:
                agreement['NotHate']+=1
                agreement_classes['NotHate']+=1
            else:
                ids_disagreed.append(k)
        else:
            total['hate'] += 1
            total_classes[v['label_str']]+=1
            if v2['label'] > 0:
                agreement['hate']+=1
            else:
                ids_disagreed.append(k)
            if v2['label']  == v['label']:
                agreement_classes[v['label_str']]+=1
    except:
        print("error")

for el in ids_disagreed:
    out_file.write(str(el) + '\n')

print("Total annotator 2")
print total
print("Agreement")
print agreement
print("------------------")
print("Total per class annotator 2")
print total_classes
print("Agreement per class")
print agreement_classes
print("------------------")
print("Total agreement:")
print(str(count - len(ids_disagreed)) + "/" + str(count))
print(str(100 * (float((count - len(ids_disagreed))) / count)))


