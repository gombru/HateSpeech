from twython import Twython, TwythonRateLimitError, TwythonError
import os
from PIL import Image
import urllib
import time

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
file = open('../../../datasets/HateSPic/Waseem/NAACL_SRW_2016.csv', 'r')

images_dir = '../../../datasets/HateSPic/Waseem/img/'
text_dir = '../../../datasets/HateSPic/Waseem/txt/'

for line in file:
    ids.append(int(line.split(',')[0]))

corrupted = 0
downloaded_images = 0
saved_txts = 0

for i,id in enumerate(ids):

    print(str(i) + " Downloaded txt: " + str(saved_txts) + " Downloaded img: " + str(downloaded_images))

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
        corrupted += 1
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
                    downloaded_images += 1

    try:
        with open(text_dir + '/' + str(t['id']) + '.txt', "w") as text_file:
            text_file.write(t['text'])
        saved_txts += 1
    except:
        print("Failed writing text")
        continue

    print "done"


print("Corrupted: " + str(corrupted))
print("Saved txt: " + str(saved_txts))
print("Downloaded images: " + str(downloaded_images))