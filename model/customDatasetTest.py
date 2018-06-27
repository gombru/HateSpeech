from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import customTransform
from PIL import Image
import json

class customDatasetTest(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, split, CenterCrop):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.CenterCrop = CenterCrop
        self.num_classes = 228


        # iMaterialist
        with open(root_dir + split + '.json', 'r') as f:
            data = json.load(f)
        num_elements = len(data["annotations"])
        # num_elements = 1000
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
            # if c % 1000 == 0: break
        print("Labels read.")


    def __len__(self):
        return len(self.indices)


    def __getitem__(self, idx):

        if self.split == '/anns/validation':
            img_name = '{}/{}/{}{}'.format(self.root_dir , 'img_val', self.indices[idx], '.jpg')
        else:
            img_name = '{}/{}/{}{}'.format(self.root_dir , 'img', self.indices[idx], '.jpg')
        try:
            image = Image.open(img_name)

        except:
            print("Img file " + img_name + " not found, using hardcoded " + img_name)
            img_name = '../../datasets/SocialMedia/img_resized_1M/cities_instagram/london/1481255189662056249.jpg'
            image = Image.open(img_name)

        if self.CenterCrop != 0:
            crop_size = 256
            image = customTransform.CenterCrop(image, crop_size, self.CenterCrop)

        im_np = np.array(image, dtype=np.float32)
        im_np = customTransform.PreprocessImage(im_np)

        out_img = np.copy(im_np)

        # Multilabel / Regression
        label = torch.from_numpy(np.array(self.labels[idx]))
        label = label.type(torch.FloatTensor)

        return torch.from_numpy(out_img), label,  self.indices[idx]