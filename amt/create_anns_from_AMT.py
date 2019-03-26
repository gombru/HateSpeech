import json

out_file = open("../../../datasets/HateSPic/MMHS/anns/5k_worker2.json","w")

# Read and iter results
results_file = open("../../../datasets/HateSPic/AMT/MMHS2/results/5k_worker2.csv","r")

results = {}
for i,line in enumerate(results_file):
    if i == 0: continue
    data=line.split(',')
    tweet_id = int(data[-3].split('/')[-1].strip("\"").strip(".jpg"))
    sentiment = data[-1]
    results[tweet_id] = sentiment

total_tweets = len(results)
not_hate = 0
racist = 0
sexist = 0
homo = 0
religion = 0
other = 0
errors = 0
anns = {}

for k,v in results.iteritems():
    if "Not" in v:
        not_hate+=1
        label = "NotHate"
        label_num = 0
    elif "Racist" in v:
        racist+=1
        label = "Racist"
        label_num = 1
    elif "Sexist" in v:
        sexist+=1
        label = "Sexist"
        label_num = 2
    elif "Homo" in v:
        homo+=1
        label = "Homophobe"
        label_num = 3
    elif "Religion" in v:
        religion+=1
        label = "Religion"
        label_num = 4
    elif "Other" in v:
        other+=1
        label = "OtherHate"
        label_num = 5

    else:
        print("Error with: " + v)
        errors+=1
        label = "Error"

    tweet_data = json.load(open("../../../datasets/HateSPic/MMHS/json/" + str(k) + ".json"))
    tweet_text = tweet_data["text"]
    tweet_url = "https://twitter.com/user/status/" + str(k)

    anns[k] = {'tweet_text': tweet_text, 'label': label_num, 'label_str': label, 'tweet_url': tweet_url, 'img_url': tweet_data['img_url']}

print("Total Tweets: " + str(total_tweets) + ". Not Hate: " + str(not_hate) + ", Racist: " + str(racist) + ", Sexist: " + str(sexist) + ", Homophobe: " + str(homo) + ", Religion: " + str(religion) + ", Other: " + str(other))
print("Errors: " + str(errors))

print("Saving results")

json.dump(anns, out_file)




