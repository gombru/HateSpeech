from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import customTransform
from PIL import Image

class customDatasetTest(Dataset):

    def __init__(self, root_dir, split, Rescale):
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
        self.hidden_state_dim = 50

        # Count number of elements
        num_elements = sum(1 for line in open(root_dir + 'tweet_embeddings/' + split))
        num_elements += sum(1 for line in open(root_dir + 'tweet_embeddings/' + split.replace('hate','nothate')))
        print("Number of elements in " + split + " (and not hate): " + str(num_elements))

        # Initialize containers
        self.tweet_ids = np.empty(num_elements, dtype="S50")
        self.labels = []
        self.tweets = np.zeros((num_elements, self.hidden_state_dim), dtype=np.float32)
        self.img_texts = np.zeros((num_elements, self.hidden_state_dim), dtype=np.float32)

        # Read image text embeddings
        img_txt_embeddings = {}
        for i, line in enumerate(open(root_dir + 'img_txt_embeddings/lstm_embeddings_img_text.txt')):
            data_img_text = line.split(',')
            embedding = np.zeros(self.hidden_state_dim)
            for c in range(self.hidden_state_dim):
                embedding[c] = float(data_img_text[c+1])
            img_txt_embeddings[int(data_img_text[0])] = embedding
        print("Img text embeddings read. Total elements: " + str(len(img_txt_embeddings)))


        # Read Hate data
        for i,line in enumerate(open(root_dir + 'tweet_embeddings/' + split)):
            data = line.split(',')
            self.tweet_ids[i] = data[0] # id
            # if 'val' in split:
            #     print('val')
            #     self.labels.append(0)
            # else:
            self.labels.append(1) # Assign hate label
            for c in range(self.hidden_state_dim): # Read LSTM hidden state
                self.tweets[i,c] = float(data[c+1])
            # Read img_text embedding
            if data[0] in img_txt_embeddings:
                self.img_texts[i,:] = img_txt_embeddings[data[0]]
            offset = i + 1


        # Read Not Hate data
        for i, line in enumerate(open(root_dir + 'tweet_embeddings/' + split.replace('hate','nothate'))):
            data = line.split(',')
            self.tweet_ids[i + offset] = data[0] # id
            self.labels.append(0) # Assign not hate label
            for c in range(self.hidden_state_dim): # Read LSTM hidden state
                self.tweets[i + offset,c] = float(data[c+1])
            # Read img_text embedding
            if data[0] in img_txt_embeddings:
                self.img_texts[i + offset,:] = img_txt_embeddings[data[0]]

        print("Data read.")


    def __len__(self):
        return len(self.tweet_ids)


    def __getitem__(self, idx):

        img_name = '{}{}/{}{}'.format(self.root_dir, 'img_resized', self.tweet_ids[idx], '.jpg')

        try:
            image = Image.open(img_name)
        except:
            img_name = '../../../datasets/HateSPic/HateSPic/img_resized/1011278006608912384.jpg'
            print("Img file " + img_name + " not found, using hardcoded " + img_name)
            image = Image.open(img_name)

        try:
            image = customTransform.Rescale(image, self.Rescale)
            im_np = np.array(image, dtype=np.float32)
            im_np = customTransform.PreprocessImage(im_np)

        except:
            img_name = '../../../datasets/HateSPic/HateSPic/img_resized/1011278006608912384.jpg'
            print("Error on data aumentation, using hardcoded: " + img_name)
            image = Image.open(img_name)
            image = customTransform.Rescale(image, self.Rescale)
            im_np = np.array(image, dtype=np.float32)
            im_np = customTransform.PreprocessImage(im_np)

        out_img = np.copy(im_np)

        # Simple Classification (class index)
        label = torch.from_numpy(np.array([int(self.labels[idx])]))
        label = label.type(torch.LongTensor)

        # Set text embedding to 0!
        self.img_texts[idx] = np.zeros(self.hidden_state_dim)
        self.tweets[idx] = np.zeros(self.hidden_state_dim)

        # Set image to 0!
        # out_img = np.zeros((3, 299, 299), dtype=np.float32)

        # Multilabel / Regression
        img_text = torch.from_numpy(np.array(self.img_texts[idx]))
        tweet = torch.from_numpy(np.array(self.tweets[idx]))
        # print(out_img.shape)

        return self.tweet_ids[idx], torch.from_numpy(out_img), img_text, tweet, label