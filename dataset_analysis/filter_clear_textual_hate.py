terms = {}
num_discarded_terms = 0
num_remaining_terms = 0
terms_without_posts = 0
remaining_terms = {}
dicarded_terms = {}
num_tweets_50k = 0
for line in open('../../../datasets/HateSPic/AMT/50K/tweetsPerTerm_50k.txt','r'):
    word = line.split('-')[0]
    line = line.replace(',','').replace('  ',' ').replace('  ',' ').replace('\n','')
    d = line.split('-')[1].split(' ')
    NotHateVotes = int(d[-2])
    HateVotes_MinusNoOther = int(d[2]) + int(d[4])+ int(d[8])+ int(d[10])
    if HateVotes_MinusNoOther*0.2 > NotHateVotes  or NotHateVotes*0.2 > HateVotes_MinusNoOther:
        dicarded_terms[word] = [d[2],d[4],d[6],d[8],d[10],d[12],d[14]]
        num_discarded_terms+=1
    elif NotHateVotes + HateVotes_MinusNoOther == 0:
        terms_without_posts+=1
    else:
        num_remaining_terms+=1
        remaining_terms[word] = [d[2],d[4],d[6],d[8],d[10],d[12],d[14]]


print('Num Remaining Terms: ' + str(num_remaining_terms) + ' Num Discarded Terms: ' + str(num_discarded_terms) + ' Terms Without posts: ' + str(terms_without_posts))
out_file = open('../../../datasets/HateSPic/AMT/50K/discardedTerms_tweetsPerTerm_50k.txt','w')
for k,d in dicarded_terms.iteritems():
    out_file.write(k + ' - ' + 'Sexist: ' + d[0] + ', Religion: ' + d[1] + ', Other: ' + d[2] + ', Racist: ' + d[3] + ', Homphobe: ' + d[4] + ', Total: ' + d[5]  + ', NotHate: ' + d[6] + '\n')
out_file.close()

out_file = open('../../../datasets/HateSPic/AMT/50K/RemainingTerms_tweetsPerTerm_50k.txt','w')
for k,d in remaining_terms.iteritems():
    num_tweets_50k += int(d[5])
    out_file.write(k + ' - ' + 'Sexist: ' + d[0] + ', Religion: ' + d[1] + ', Other: ' + d[2] + ', Racist: ' + d[3] + ', Homphobe: ' + d[4] + ', Total: ' + d[5]  + ', NotHate: ' + d[6] + '\n')
out_file.close()

print("Tweets that would be accepted form the 50k: " + str(num_tweets_50k))
