from preprocess_tweets import tweet_preprocessing
import random

base_path = '../../../../datasets/HateSPic/Zhang/'
out_path = '../../../../datasets/HateSPic/lstm_data/annotated/'
datasets = ['dt/labeled_data_all_2classes_only.csv','rm/labeled_data_tweets_only.csv','wz-ls/labeled_data_text.csv']

hate_file = open(out_path + 'tweets.hate', 'w')
nothate_file = open(out_path + 'tweets.nothate', 'w')

hate_file_val = open(out_path + 'val_tweets.hate', 'w')
nothate_file_val = open(out_path + 'val_tweets.nothate', 'w')

for i, dataset in enumerate(datasets):
    with open(base_path + dataset, 'r') as f:
        for line_num, line in enumerate(f):

            try:

                if line_num == 0: continue

                if i == 0:
                    if len(line.split(',')) < 6:
                        #print("Continuing: " + line)
                        continue
                    text = ' '.join(line.split(',')[6:])
                    label_id = int(line.split(',')[5])

                if i in[1,2]:
                    if len(line.split(',')) < 2:
                        #print("Continuing: " + line)
                        continue
                    text = ' '.join(line.split(',')[1:])
                    label_id = int(line.split(',')[0])

                text = tweet_preprocessing(text)

                # Discard short tweets
                if len(text) < 5: continue
                if len(text.split(' ')) < 3: continue


                if label_id == 0:
                    if random.randint(0,10) == 0:
                        hate_file_val.write(text)
                    else:
                        hate_file.write(text)

                if label_id == 2:
                    if random.randint(0, 62) == 0:
                        nothate_file_val.write(text)
                    else:
                        nothate_file.write(text)

            except:
                print("Error with line: " + line)
                continue

hate_file.close()
nothate_file.close()
print "DONE"