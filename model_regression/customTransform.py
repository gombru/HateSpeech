import torch
from skimage import io, transform
import numpy as np
from PIL import Image, ImageOps
import random


def PreprocessImage(im):
    mean = (104.00698793, 116.66876762, 122.67891434)
    if (im.shape.__len__() < 3 or im.shape[2] > 3):
        im_gray = im
        im = Image.new("RGB", im_gray.size)
        im.paste(im_gray)
        im = np.array(im, dtype=np.float32)
    im = im[:, :, ::-1]
    im -= mean
    im = im.transpose((2, 0, 1))
    return im

def Mirror(im):
    return ImageOps.mirror(im)

def Rescale(im, output_size):
    if im.size[0] != output_size or im.size[1] != output_size:
        im = im.resize((output_size, output_size), Image.ANTIALIAS)
    return im


def RandomCrop(im, output_size):

    # Handle small images
    try:
        width, height = im.size
        if width < output_size -1:
            im = im.resize((width * 2, height * 2), Image.ANTIALIAS)

        if height < output_size -1:
            im = im.resize((width * 2, height * 2), Image.ANTIALIAS)

        width, height = im.size
        left = random.randint(0, width - output_size - 1)
        top = random.randint(0, height - output_size - 1)
        im = im.crop((left, top, left + output_size, top + output_size))
    except:
        im = Rescale(im,output_size)
    return im

def CenterCrop(im, crop_size, output_size):
    width, height = im.size
    if width != crop_size:

        left = (width - crop_size) / 2
        right = width - left
        im = im.crop((left, 0, right, height))

    if height != crop_size:
        top = (height - crop_size) / 2
        bot = height - top
        im = im.crop((0, top, width, bot))

    return im.resize((output_size, output_size), Image.ANTIALIAS)
