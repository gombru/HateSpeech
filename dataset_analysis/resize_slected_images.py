import glob
from PIL import Image
from joblib import Parallel, delayed
import os


images_path = '../../../datasets/HateSPic/MMHS/img_extra/'
im_dest_path = '../../../datasets/HateSPic/MMHS/img_resized/'
selected_anns_path = '../../../datasets/HateSPic/AMT/MMHS2/2label_extra/'

minSize = 500

def resize(json_path):
    try:
        tweet_id = json_path.split('/')[-1].replace('.json','')
        im_path = images_path + tweet_id + '.jpg'

        if os.path.exists(im_dest_path + im_path.split('/')[-1]):
            # print("File exists, skipping")
            return

        im = Image.open(im_path)

        w = im.size[0]
        h = im.size[1]

        if w < h:
            new_width = minSize
            new_height = int(minSize * (float(h) / w))

        if h <= w:
            new_height = minSize
            new_width = int(minSize * (float(w) / h))

        im = im.resize((new_width, new_height), Image.ANTIALIAS)
        im.save(im_dest_path + im_path.split('/')[-1])

    except:
        print "Failed resizing image"
        return


if not os.path.exists(im_dest_path):
    os.makedirs(im_dest_path)
Parallel(n_jobs=4)(delayed(resize)(file) for file in glob.glob(selected_anns_path + "/*.json"))

print "DONE"