import re
import json

out_file = open("../../../datasets/HateSPic/AMT/results/tweetsPerTerm_5k_1.txt","w")

# Read Search terms
terms_file = open("../twitter/termsAndMine.txt","r")
tweets_per_term = {}
for line in terms_file:
    line=re.sub('\r|\n|\t|#','',line).lower()
    tweets_per_term[line] = {"Total": 0, "NotHate": 0, "Racist": 0, "Sexist": 0, "Homophobe": 0, "Religion": 0, "Other": 0}


def find_searchWord(tweet_text):
    for k in tweets_per_term.keys():
        if k in tweet_text:
            return k
    return"NoKeyWordFound"

# Read and iter results
results_file = open("../../../datasets/HateSPic/AMT/results/5k_1.csv","r")

results = {}
for i,line in enumerate(results_file):
    if i == 0: continue
    data=line.split(',')
    tweet_id = int(data[-2].split('/')[-1].strip("\""))
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
for k,v in results.iteritems():
    if "Not" in v:
        not_hate+=1
        label = "NotHate"
    elif "Racist" in v:
        racist+=1
        label = "Racist"
    elif "Sexist" in v:
        sexist+=1
        label = "Sexist"
    elif "Homo" in v:
        homo+=1
        label = "Homophobe"
    elif "Religion" in v:
        religion+=1
        label = "Religion"
    elif "Other" in v:
        other+=1
        label = "Other"
    else:
        print("Error with: " + v)
        errors+=1
        label = "Error"

    tweet_data = json.load(open("../../../datasets/HateSPic/AMT/2label/" + str(k) + ".json"))
    tweet_text = tweet_data["text"].strip("-").strip("#").lower()
    search_word = find_searchWord(tweet_text)
    if search_word == "NoKeyWordFound":
        print("Error: NoKeyWordFound")
        print(tweet_text)
    else:
        tweets_per_term[search_word]["Total"] += 1
        tweets_per_term[search_word][label] += 1

print("Total Tweets: " + str(total_tweets) + ". Not Hate: " + str(not_hate) + ", Racist: " + str(racist) + ", Sexist: " + str(sexist) + ", Homophobe: " + str(homo) + ", Religion: " + str(religion) + ", Other: " + str(other))
print("Errors: " + str(errors))

print("Saving results")

for k,v in tweets_per_term.iteritems():
    out_file.write(k + ' - ')
    for type,count in v.iteritems():
        out_file.write(type + ": " + str(count) + ', ')
    out_file.write('\n')





