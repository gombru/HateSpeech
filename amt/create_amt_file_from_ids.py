import random

ids_file_path = '../../../datasets/HateSPic/AMT/ids_30_amt_test.csv'
out_file_path = '../../../datasets/HateSPic/AMT/30_amt_test.csv'


ids = open(ids_file_path).read().split("\n")
num_ids = []
for id in ids:
    if len(id) > 10:
        num_ids.append(int(id))

random.shuffle(num_ids)
out_file = open(out_file_path,'w')
out_file.write('tweet_url\n')

for id in num_ids:
    out_file.write("https://twitter.com/user/status/" + str(id) + '\n')