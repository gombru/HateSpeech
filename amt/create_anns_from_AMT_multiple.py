import json

out_file = open("../../../datasets/HateSPic/MMHS/anns/MMHS150K_GT.json","w")

anns = json.load(open("../../../datasets/HateSPic/MMHS/anns/old/50k_3workers.json","r"))
print("Loaded old anns: " + str(len(anns)))

# Read and iter results
results_file = open("../../../datasets/HateSPic/AMT/MMHS2/results/100k_final.csv","r")

results = {}
rejected = 0
for i,line in enumerate(results_file):
    if i == 0: continue
    data=line.split(',')
    if "Rejected" in data[-15]:
        rejected += 1
        continue
    if len(data) < 2: continue
    if len(data) == 33:
        tweet_id = int(data[-2].split('/')[-1].strip("\"").strip(".jpg"))
    else:
        tweet_id = int(data[-3].split('/')[-1].strip("\"").strip(".jpg"))
    sentiment = data[-1]
    if tweet_id not in results:
        results[tweet_id] = [sentiment]
    else:
        results[tweet_id].append(sentiment)

print("Rejected: " + str(rejected))
print("Accepted: " + str(len(results)))

total_tweets = len(results)
not_hate = 0
racist = 0
sexist = 0
homo = 0
religion = 0
other = 0
errors = 0
# anns = {}

majority_not_hate = 0
majority_hate = 0
majority_racist = 0
majority_sexist = 0
majority_homo = 0
majority_religion = 0
majority_other = 0

for k,v in results.iteritems():
    labels = []
    label_num = []
    for label in v:
        if "Not" in label:
            not_hate+=1
            labels.append("NotHate")
            label_num.append(0)
        elif "Racist" in label:
            racist+=1
            labels.append("Racist")
            label_num.append(1)
        elif "Sexist" in label:
            sexist+=1
            labels.append("Sexist")
            label_num.append(2)
        elif "Homo" in label:
            homo+=1
            labels.append("Homophobe")
            label_num.append(3)
        elif "Religion" in label:
            religion+=1
            labels.append("Religion")
            label_num.append(4)
        elif "Other" in label:
            other+=1
            labels.append("OtherHate")
            label_num.append(5)

        else:
            print("Error with: " + v)
            errors+=1
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


    tweet_data = json.load(open("../../../datasets/HateSPic/MMHS/json/" + str(k) + ".json"))
    tweet_text = tweet_data["text"]
    tweet_url = "https://twitter.com/user/status/" + str(k)

    anns[k] = {'tweet_text': tweet_text, 'labels': label_num, 'labels_str': labels, 'tweet_url': tweet_url, 'img_url': tweet_data['img_url']}
    # print(anns[k])
print("Total Tweets Votes: " + str(total_tweets) + ". Not Hate: " + str(not_hate) + ", Racist: " + str(racist) + ", Sexist: " + str(sexist) + ", Homophobe: " + str(homo) + ", Religion: " + str(religion) + ", Other: " + str(other))
print("Total Tweets Mojority Voting: Not Hate: " + str(majority_not_hate) + ", Hate: " + str(majority_hate) + ", Racist: " + str(majority_racist) + ", Sexist: " + str(majority_sexist) + ", Homophobe: " + str(majority_homo) + ", Religion: " + str(majority_religion) + ", Other: " + str(majority_other))
print("Errors: " + str(errors))
print(len(anns))
print("Saving results")

json.dump(anns, out_file)




