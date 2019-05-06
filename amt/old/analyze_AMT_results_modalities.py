import re
import json


out_file = open("../../../datasets/HateSPic/MMHS/anns/1k_3workers_v2.json","w")


# Read and iter results
results_file = open("../../../datasets/HateSPic/AMT/MMHS2/results/1k_3workers_v2.csv","r")

results = {}
for i,line in enumerate(results_file):
    if i == 0: continue
    data=line.split(',')
    tweet_id = int(data[-4].split('/')[-1].split('.')[0])
    sentiment = data[-1]
    modality = data[-2]
    if tweet_id not in results:
        results[tweet_id] = {}
        results[tweet_id]['sentiment'] = [sentiment]
        results[tweet_id]['modality'] = [modality]
    else:
        results[tweet_id]['sentiment'].append(sentiment)
        results[tweet_id]['modality'].append(modality)

anns = {}
total_tweets = len(results)
not_hate = 0
racist = 0
sexist = 0
homo = 0
religion = 0
other = 0
errors = 0

not_hate_mod = 0
text = 0
image = 0
both = 0
combined = 0

for k,v in results.iteritems():

    labels = []
    modalities = []
    modalities_num = []
    label_num = []

    sentiment = v['sentiment']
    modality = v['modality']

    print sentiment
    print modality

    for label in sentiment:
        if "Not" in label:
            not_hate += 1
            labels.append("NotHate")
            label_num.append(0)
        elif "Racist" in label:
            racist += 1
            labels.append("Racist")
            label_num.append(1)
        elif "Sexist" in label:
            sexist += 1
            labels.append("Sexist")
            label_num.append(2)
        elif "Homo" in label:
            homo += 1
            labels.append("Homophobe")
            label_num.append(3)
        elif "Religion" in label:
            religion += 1
            labels.append("Religion")
            label_num.append(4)
        elif "Other" in label:
            other += 1
            labels.append("OtherHate")
            label_num.append(5)

    for label in modality:
        if "Not" in label:
            not_hate_mod+=1
            modalities.append("NotHate")
            modalities_num.append(0)
        elif "Text" in label:
            text+=1
            modalities.append("Text")
            modalities_num.append(1)
        elif "Image" in label:
            image+=1
            modalities.append("Image")
            modalities_num.append(2)
        elif "Both" in label:
            both+=1
            modalities.append("Both")
            modalities_num.append(3)
        elif "Combined" in label:
            combined+=1
            modalities.append("Combined")
            modalities_num.append(4)
        else:
            print("Error with: " + str(k))
            errors+=1
            label = "Error"

    tweet_data = json.load(open("../../../datasets/HateSPic/MMHS/json/" + str(k) + ".json"))
    tweet_text = tweet_data["text"]
    tweet_url = "https://twitter.com/user/status/" + str(k)

    anns[k] = {'tweet_text': tweet_text, 'labels': label_num, 'labels_str': labels, 'modalities': modalities_num, 'modalities_str': modalities, 'tweet_url': tweet_url, 'img_url': tweet_data['img_url']}

print("Total Tweets: " + str(total_tweets) + ". Not Hate: " + str(not_hate) + ", Racist: " + str(racist) + ", Sexist: " + str(sexist) + ", Homophobe: " + str(homo) + ", Religion: " + str(religion) + ", Other: " + str(other))
print("Not Hate: " + str(not_hate_mod) + ", Text: " + str(text) + ", Image: " + str(image) + ", Both: " + str(both) + ", Combined: " + str(combined))

print("Errors: " + str(errors))

json.dump(anns, out_file)





