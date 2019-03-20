import re
import os
import random
import codecs
from torchtext import data

# def clean_str(string):
#     """
#     Tokenization/string cleaning for all datasets except for SST.
#     Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
#     """
#     string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
#     string = re.sub(r"\'s", " \'s", string)
#     string = re.sub(r"\'ve", " \'ve", string)
#     string = re.sub(r"n\'t", " n\'t", string)
#     string = re.sub(r"\'re", " \'re", string)
#     string = re.sub(r"\'d", " \'d", string)
#     string = re.sub(r"\'ll", " \'ll", string)
#     string = re.sub(r",", " , ", string)
#     string = re.sub(r"!", " ! ", string)
#     string = re.sub(r"\(", " \( ", string)
#     string = re.sub(r"\)", " \) ", string)
#     string = re.sub(r"\?", " \? ", string)
#     string = re.sub(r"\s{2,}", " ", string)
#     return string.strip()


class HD(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, img_txts=None, path=None, examples=None, split=None, **kwargs):
        """Create an Emotion Dataset instance given a path and fields.
        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        # text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]
        if examples is None:
            path = self.dirname if path is None else path
            examples = []

            if split == 'train':
                with codecs.open(os.path.join(path, 'tweets.train_hate'),'r','utf8') as f:
                    for line in f:
                        if int(line.split(',')[0]) in img_txts:
                            words = line.split(',')[1] + ' image ' + line.split(',')[0] + ' text ' + img_txts[int(line.split(',')[0])]
                        else:
                            words = line.split(',')[1] + ' image ' + line.split(',')[0]
                        examples.append(data.Example.fromlist([words,'hate'], fields))

                with codecs.open(os.path.join(path, 'tweets.train_nothate'),'r','utf8') as f:
                    for line in f:
                        if int(line.split(',')[0]) in img_txts:
                            words = line.split(',')[1] + ' image ' + line.split(',')[0] + ' text ' + img_txts[
                                int(line.split(',')[0])]
                        else:
                            words = line.split(',')[1] + ' image ' + line.split(',')[0]
                        examples.append(data.Example.fromlist([words, 'nothate'], fields))

            if split == 'val':
                with codecs.open(os.path.join(path, 'tweets.val_hate'), 'r', 'utf8') as f:
                    for line in f:
                        if int(line.split(',')[0]) in img_txts:
                            words = line.split(',')[1] + ' image ' + line.split(',')[0] + ' text ' + img_txts[
                                int(line.split(',')[0])]
                        else:
                            words = line.split(',')[1] + ' image ' + line.split(',')[0]
                        examples.append(data.Example.fromlist([words, 'hate'], fields))

                with codecs.open(os.path.join(path, 'tweets.val_nothate'), 'r', 'utf8') as f:
                    for line in f:
                        if int(line.split(',')[0]) in img_txts:
                            words = line.split(',')[1] + ' image ' + line.split(',')[0] + ' text ' + img_txts[
                                int(line.split(',')[0])]
                        else:
                            words = line.split(',')[1] + ' image ' + line.split(',')[0]
                        examples.append(data.Example.fromlist([words, 'nothate'], fields))

            if split == 'all_img_ids':
                all_img_ids = ''
                image_features_path = '../../../datasets/HateSPic/HateSPic/img_embeddings/MMHSv3mm_ImageEmbeddings200_ALL_ADAM_bs32_lrMMe6_lrCNNe7_epoch_160_ValAcc_62_ValLoss_0.65.txt'
                for line in open(image_features_path, 'r'):
                    all_img_ids += line.split(',')[0] + ' '
                examples.append(data.Example.fromlist([all_img_ids, 'hate'], fields))


        super(HD, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, shuffle=True ,root='.',path="../../../datasets/HateSPic/lstm_data/HateSPic_v3mm/", **kwargs):
        """Create dataset objects for splits of the MR dataset.
        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """

        # LOAD image texts
        img_txt_path = '../../../datasets/HateSPic/lstm_data/img_txt/tweets.img_text'
        img_txts = {}
        for line in open(img_txt_path,'r'):
            id = int(line.strip(',')[0])
            text = line.strip(',')[1].replace('\n', '').replace('\r', '')
            img_txts[id] = text

        # LOAD img_ids vocab to have them in vocab
        all_img_ids_examples = cls(text_field, label_field, img_txts = img_txts, path=path, split='all_img_ids', **kwargs).examples

        train_examples = cls(text_field, label_field, img_txts = img_txts, path=path, split='train', **kwargs).examples
        if shuffle: random.shuffle(train_examples)

        dev_examples = cls(text_field, label_field, img_txts = img_txts, path=path, split='val', **kwargs).examples
        if shuffle: random.shuffle(dev_examples)

        # dev_index = int(len(examples) - 0.05 * len(examples))
        # train_examples = examples[0:dev_index]
        # dev_examples = examples[dev_index:]
        # random.shuffle(train_examples)
        # random.shuffle(dev_examples)

        print('train:',len(train_examples),'dev:',len(dev_examples))
        return cls(text_field, label_field, examples=train_examples), cls(text_field, label_field, examples=dev_examples), cls(text_field, label_field, examples=all_img_ids_examples)

def load_HD(text_field, label_field, batch_size):
    print('loading data')
    train_data, dev_data, all_img_ids_data = HD.splits(text_field, label_field)
    text_field.build_vocab(train_data, dev_data, all_img_ids_data)
    print("Img ids added to vocab")
    label_field.build_vocab(train_data, dev_data)
    print('Size vocab: ' + str(len(text_field.vocab.itos)))

    print('building batches')
    train_iter, dev_iter = data.Iterator.splits(
        (train_data, dev_data), batch_sizes=(batch_size, batch_size),repeat=False, shuffle=False,
        device = -1
    )

    return train_iter, dev_iter