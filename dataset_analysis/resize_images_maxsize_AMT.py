import glob
from PIL import Image
from joblib import Parallel, delayed
import os


images_path = '/home/rgomez/datasets/HateSPic/MMHS50K/img_resized/'
im_dest_path = '/home/rgomez/datasets/HateSPic/MMHS50K/img_resized_amt/'

maxSize = 400

def resize(im_path):
    try:
        im = Image.open(im_path)

        w = im.size[0]
        h = im.size[1]

        if w < h:
            new_height = maxSize
            new_width = int(maxSize * (float(w) / h))

        if h <= w:
            new_width = maxSize
            new_height = int(maxSize * (float(h) / w))

        im = im.resize((new_width, new_height), Image.ANTIALIAS)
        im.save(im_dest_path + im_path.split('/')[-1])

    except:
        print "Failed resizing image"
        return


if not os.path.exists(im_dest_path):
    os.makedirs(im_dest_path)
Parallel(n_jobs=8)(delayed(resize)(file) for file in glob.glob(images_path + "/*.jpeg"))

print "DONE"