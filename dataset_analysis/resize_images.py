import glob
from PIL import Image
from joblib import Parallel, delayed
import os


images_path = '../../../datasets/HateSPic/HateSPic/img/'
im_dest_path = '../../../datasets/HateSPic/HateSPic/img_resized/'

minSize = 500

def resize(im_path):
    try:
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
Parallel(n_jobs=12)(delayed(resize)(file) for file in glob.glob(images_path + "/*.jpg"))

print "DONE"