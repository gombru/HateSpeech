from twython import Twython, TwythonRateLimitError, TwythonError
import os
from PIL import Image
import urllib
import time
import json
import random

CONSUMER_KEY = "uHmr7pmSU6yBiEtbpZQPSsqlQ"
CONSUMER_SECRET = "xICgXtFxp6HrQDQh2oAd6OysFxDoO9mo5blarLeBB8aegALrkH"
OAUTH_TOKEN = "81841533-PU84e9z6jNt1AtgHP13GnS8tRJGTMSJ3lLvMevYpE"
OAUTH_TOKEN_SECRET = "33ySOzqucOiMCst5dZcRcbyzKPjKx7xSNp9aj7esdCFa5"
twitter = Twython(CONSUMER_KEY, CONSUMER_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)


def download_save_image(url, image_path):
    resource = urllib.urlopen(url)
    output = open(image_path, "wb")
    output.write(resource.read())
    output.close()

ids = []
file = open('../../../datasets/HateSPic/hate_speech_icwsm18/indices.csv', 'r')
dataset_name = 'SemiSupervised'

out_dir = '../../../datasets/HateSPic/HateSPicLabeler/original_json/' + dataset_name + '/'
images_dir = '../../../datasets/HateSPic/hate_speech_icwsm18/img/'


for line in file:
    ids.append(int(line.split(',')[0]))
random.shuffle(ids)

created = 0

for i,id in enumerate(ids):

    print(str(i) + " Created: " + str(created))

    try:
        t = twitter.show_status(id=id)

    except TwythonRateLimitError as error:
        remainder = float(twitter.get_lastfunction_header(header='x-rate-limit-reset')) - time.time()
        print("Rate limit reched, sleeping (s): " + str(remainder))
        del twitter
        if remainder <= 0: remainder = 1
        time.sleep(remainder)
        twitter = Twython(CONSUMER_KEY, CONSUMER_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
        print("Resuming -->")
        continue

    except TwythonError as error:
        print "Tweet does not exist: " + str(id)
        continue

    # Check if tweet has image
    if t.has_key(u'entities'):
        if t['entities'].has_key(u'media'):
            if t['entities']['media'][0]['type'] == u'photo':
                if t['entities']['media'][0]['media_url'][-3:] == 'jpg':

                    image_path = images_dir + '/' + str(t['id']) + ".jpg"

                    # Download image
                    try:
                        download_save_image(t['entities']['media'][0]['media_url'], image_path)
                        # Check image can be opened
                        im = Image.open(image_path)
                    except:
                        if os.path.exists(image_path):
                            os.remove(image_path)  # Remove the corrupted file
                        print "Failed downloading image from: " + t['entities']['media'][0]['media_url']
                        continue

                try:
                    label_id = None
                    # Get original class

                    # Can't download TM tweets, they don't have ID!
                    # if dataset_name == 'DT':
                    #     if len(line.split(',')) < 6:
                    #         continue
                    #     label_id = int(line.split(',')[5])

                    if dataset_name == 'WZ-LS':
                        if len(line.split(',')) < 2:
                            continue
                        label_id = int(line.split(',')[1])

                    # Can't download TM tweets, they don't have ID!
                    # if dataset_name == 'RM':
                    #     if len(line.split(',')) < 2:
                    #         continue
                    #     label_id = int(line.split(',')[0])

                    if label_id == 0: label_id = 1 # Hate
                    elif label_id == 2: label_id = 0 # Not hate

                    if dataset_name == 'SemiSupervised':
                        label_id = 1

                    if label_id not in [0,1]:
                        print("WARNIGN WRONG LABEL ID")

                    info = {}
                    info['id'] = t['id']
                    info['img_url'] = t['entities']['media'][0]['media_url']
                    info['text'] = t['text'].encode("utf8", "ignore").replace('\n', ' ').replace('\r', ' ')
                    info['dataset'] = dataset_name
                    info['hate_votes'] = 0
                    info['not_hate_votes'] = 0
                    info['voters'] = 0
                    info['HateSPiclabeler_annotation'] = None
                    info['original_annotation'] = label_id

                    with open(out_dir + '/' + str(t['id']) + '.json', "w") as out_file:
                        json.dump(info, out_file)
                    created += 1

                except:
                    print("Failed writing")
                    continue

    print "done"


print("created: " + str(created))