from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import customTransform
from PIL import Image
import json

class CustomDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, split, Rescale, RandomCrop, Mirror):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.Rescale = Rescale
        self.RandomCrop = RandomCrop
        self.Mirror = Mirror
        self.num_classes = 228

        # split_f  = '{}/{}.txt'.format(root_dir, split)
        # self.indices = open(split_f, 'r').read().splitlines()

        # Simple Classification (class index)
        # self.labels = [int(i.split(',', 1)[1]) for i in self.indices]

        # Multilabel / Regression
        # self.labels = []
        # num_classes = 5
        # for i in self.indices:
        #     data = i.split(',')
        #     cur_label = np.zeros(num_classes)
        #     for c in range(0, num_classes):
        #         cur_label[c] = float(data[c+1])
        #     self.labels.append(cur_label)

        # Save image names
        # self.indices = [i.split(',', 1)[0] for i in self.indices]
        # print("Num images in " + split + " --> " + str(len(self.indices)))

        # iMaterialist
        with open(root_dir + split + '.json', 'r') as f:
            data = json.load(f)
        num_elements = len(data["annotations"])
        #num_elements = 9001
        print("Number of images: " + str(num_elements))

        # Load labels for multiclass
        self.indices = np.empty([num_elements], dtype="S50")
        self.labels = np.zeros((num_elements, self.num_classes), dtype=np.float32)

        for c, image in enumerate(data["annotations"]):
            gt_labels = image["labelId"]
            self.indices[c] = image["imageId"]
            for l in gt_labels:
                self.labels[c, int(l) - 1] = 1
            if c % 100000 == 0: print("Read " + str(c) + " / " + str(num_elements))
            #if c == 9000: break
        print("Labels read.")


    def __len__(self):
        return len(self.indices)


    def __getitem__(self, idx):
        # img_name = self.root_dir + '/img_resized_1M/cities_instagram/' + self.indices[idx] + '.jpg'
        if self.split == '/anns/validation':
            img_name = '{}/{}/{}{}'.format(self.root_dir , 'img_val', self.indices[idx], '.jpg')
        else:
            img_name = '{}/{}/{}{}'.format(self.root_dir , 'img', self.indices[idx], '.jpg')
        try:
            image = Image.open(img_name)
            # print("FOUND " + img_name)
        except:
            print("Img file " + img_name + " not found, using hardcoded " + img_name)
            img_name = '../../datasets/SocialMedia/img_resized_1M/cities_instagram/london/1481255189662056249.jpg'
            image = Image.open(img_name)

        try:
            width, height = image.size
            if self.RandomCrop >= width or self.RandomCrop >= height:
                image = image.resize((int(width*1.5), int(height*1.5)), Image.ANTIALIAS)

            if self.Rescale != 0:
                image = customTransform.Rescale(image,self.Rescale)

            if self.RandomCrop != 0:
                image = customTransform.RandomCrop(image,self.RandomCrop)

            if self.Mirror:
                image = customTransform.Mirror(image)

            im_np = np.array(image, dtype=np.float32)
            im_np = customTransform.PreprocessImage(im_np)

        except:
            print("Error on data aumentation, using hardcoded")
            img_name = self.root_dir + '/img_resized_1M/cities_instagram/london/1481255189662056249.jpg'
            image = Image.open(img_name)
            if self.RandomCrop != 0:
                image = customTransform.RandomCrop(image,self.RandomCrop)
            im_np = np.array(image, dtype=np.float32)
            im_np = customTransform.PreprocessImage(im_np)

        out_img = np.copy(im_np)

        # Simple Classification (class index)
        # label = torch.from_numpy(np.array([int(self.labels[idx])]))
        # label = label.type(torch.LongTensor)

        # Multilabel / Regression
        label = torch.from_numpy(np.array(self.labels[idx]))
        label = label.type(torch.FloatTensor)

        return torch.from_numpy(out_img), label