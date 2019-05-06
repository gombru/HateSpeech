import json

anns = json.load(open("../../../datasets/HateSPic/MMHS/anns/MMHS150K_GT.json","r"))
print("Loaded anns: " + str(len(anns)))

majority_not_hate = 0
majority_hate = 0
majority_racist = 0
majority_sexist = 0
majority_homo = 0
majority_religion = 0
majority_other = 0

for k,v in anns.iteritems():
    labels = []
    label_num = []
    print(len(v["labels_str"]))
    for label in v["labels_str"]:
        if "Not" in label:
            label_num.append(0)
        elif "Racist" in label:
            label_num.append(1)
        elif "Sexist" in label:
            label_num.append(2)
        elif "Homo" in label:
            label_num.append(3)
        elif "Religion" in label:
            label_num.append(4)
        elif "Other" in label:
            label_num.append(5)
        else:
            print("Error with: " + label)
            label = "Error"

    if label_num.count(0) > 1:
        majority_not_hate+=1
    else:
        majority_hate+=1
        if label_num.count(1) > 1:
            majority_racist+=1
        elif label_num.count(2) > 1:
            majority_sexist+=1
        elif label_num.count(3) > 1:
            majority_homo+=1
        elif label_num.count(4) > 1:
            majority_religion+=1
        elif label_num.count(5) > 1:
            majority_other+=1

print("Total Tweets Mojority Voting: Not Hate: " + str(majority_not_hate) + ", Hate: " + str(majority_hate) + ", Racist: " + str(majority_racist) + ", Sexist: " + str(majority_sexist) + ", Homophobe: " + str(majority_homo) + ", Religion: " + str(majority_religion) + ", Other: " + str(majority_other))




