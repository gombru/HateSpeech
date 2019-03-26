import json

out_file = open("../../../datasets/HateSPic/MMHS/anns/50k_3workers.json","w")

# Read and iter results
results_file = open("../../../datasets/HateSPic/AMT/MMHS2/results/50k_3workers.csv","r")

results = {}
for i,line in enumerate(results_file):
    # if i == 0: continue
    data=line.split(',')
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
    print(i)

total_tweets = len(results)
not_hate = 0
racist = 0
sexist = 0
homo = 0
religion = 0
other = 0
errors = 0
anns = {}
count = 0

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

    count+=1
    print(count)

    tweet_data = json.load(open("../../../datasets/HateSPic/MMHS/json/" + str(k) + ".json"))
    tweet_text = tweet_data["text"]
    tweet_url = "https://twitter.com/user/status/" + str(k)

    anns[k] = {'tweet_text': tweet_text, 'labels': label_num, 'labels_str': labels, 'tweet_url': tweet_url, 'img_url': tweet_data['img_url']}
    # print(anns[k])
print("Total Tweets: " + str(total_tweets) + ". Not Hate: " + str(not_hate) + ", Racist: " + str(racist) + ", Sexist: " + str(sexist) + ", Homophobe: " + str(homo) + ", Religion: " + str(religion) + ", Other: " + str(other))
print("Errors: " + str(errors))

print("Saving results")

json.dump(anns, out_file)




