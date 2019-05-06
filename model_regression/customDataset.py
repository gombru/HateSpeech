from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import customTransform
from PIL import Image

class CustomDataset(Dataset):

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
        self.hidden_state_dim = 150

        # Count number of elements
        num_elements = sum(1 for line in open(root_dir + 'tweet_embeddings/' + split))
        print("Number of elements in " + split + ": " + str(num_elements))

        # Initialize containers
        self.tweet_ids = np.empty(num_elements, dtype="S50")
        self.labels = np.empty(num_elements, dtype=np.float32)
        self.tweets = np.zeros((num_elements, self.hidden_state_dim), dtype=np.float32)
        self.img_texts = np.zeros((num_elements, self.hidden_state_dim), dtype=np.float32)

        # Read image text embeddings
        img_txt_embeddings = {}
        for i, line in enumerate(open(root_dir + 'tweet_embeddings/MMHS_lstm_embeddings_regression/img_txt.txt')):
            data_img_text = line.split(',')
            embedding = np.zeros(self.hidden_state_dim)
            for c in range(self.hidden_state_dim):
                embedding[c] = float(data_img_text[c+2])
            img_txt_embeddings[int(data_img_text[0])] = embedding
        print("Img text embeddings read. Total elements: " + str(len(img_txt_embeddings)))


        # Read data
        for i,line in enumerate(open(root_dir + 'tweet_embeddings/' + split)):
            data = line.split(',')
            self.tweet_ids[i] = data[0]
            self.labels[i] = data[1]
            for c in range(self.hidden_state_dim): # Read LSTM hidden state
                self.tweets[i,c] = float(data[c+2])
            # Read img_text embedding
            if data[0] in img_txt_embeddings:
                self.img_texts[i,:] = img_txt_embeddings[data[0]]

        print("Data read.")


    def __len__(self):
        return len(self.tweet_ids)


    def __getitem__(self, idx):
        img_name = '{}{}/{}{}'.format(self.root_dir, 'img_resized', self.tweet_ids[idx],'.jpg')#, '.jpg')

        try:
            image = Image.open(img_name)
            # print("FOUND " + img_name)
        except:
            new_img_name = '../../../datasets/HateSPic/MMHS/img_resized/1037385299310112768.jpg'
            print("Img file " + img_name + " not found, using hardcoded " + new_img_name)
            image = Image.open(new_img_name)

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
            print("Error in data aumentation with image " + img_name)
            img_name = '../../../datasets/HateSPic/MMHS/img_resized/1037385299310112768.jpg'
            print("Using hardcoded: " + img_name)
            image = Image.open(img_name)
            if self.RandomCrop != 0:
                image = customTransform.RandomCrop(image,self.RandomCrop)
            if self.Rescale != 0:
                image = customTransform.Rescale(image,self.Rescale)
            im_np = np.array(image, dtype=np.float32)
            im_np = customTransform.PreprocessImage(im_np)

        out_img = np.copy(im_np)

        # Multilabel / Regression
        img_text = torch.from_numpy(np.array(self.img_texts[idx]))
        label = torch.from_numpy(np.array(self.labels[idx]))
        tweet = torch.from_numpy(np.array(self.tweets[idx]))
        # print(out_img.shape)

        return torch.from_numpy(out_img), img_text, tweet, label