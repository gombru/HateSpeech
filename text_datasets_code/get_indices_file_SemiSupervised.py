import os

out_file = open('../../../datasets/HateSPic/hate_speech_icwsm18/indices.csv','w')

for file in os.listdir('../../../datasets/HateSPic/hate_speech_icwsm18/twitter_key_phrase_based_datasets/'):
    for i,line in enumerate(open('../../../datasets/HateSPic/hate_speech_icwsm18/twitter_key_phrase_based_datasets/' + file,'r')):
        if i == 0: continue
        out_file.write(str(int(line)) + ',' + '1' + '\n') #1 is hate

