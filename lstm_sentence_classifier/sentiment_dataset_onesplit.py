import re
import os
import random
import tarfile
import codecs
from torchtext import data
SEED = 1

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


class ED(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, id_field, path=None, examples=None, **kwargs):
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
        fields = [('text', text_field), ('label', label_field), ('id', id_field)]
        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            with codecs.open(os.path.join(path, 'captions.amusement'),'r','utf8') as f:
                examples += [
                    data.Example.fromlist([line.split(',',1)[1], 'amusement', line.split(',')[0]], fields) for line in f]
            with codecs.open(os.path.join(path, 'captions.anger'),'r','utf8') as f:
                examples += [
                    data.Example.fromlist([line.split(',',1)[1], 'anger', line.split(',')[0]], fields) for line in f]
            with codecs.open(os.path.join(path, 'captions.awe'),'r','utf8') as f:
                examples += [
                    data.Example.fromlist([line.split(',',1)[1], 'awe', line.split(',')[0]], fields) for line in f]
            with codecs.open(os.path.join(path, 'captions.contentment'),'r','utf8') as f:
                examples += [
                    data.Example.fromlist([line.split(',',1)[1], 'contentment', line.split(',')[0]], fields) for line in f]
            with codecs.open(os.path.join(path, 'captions.disgust'),'r','utf8') as f:
                examples += [
                    data.Example.fromlist([line.split(',',1)[1], 'disgust', line.split(',')[0]], fields) for line in f]
            with codecs.open(os.path.join(path, 'captions.excitement'),'r','utf8') as f:
                examples += [
                    data.Example.fromlist([line.split(',',1)[1], 'excitement', line.split(',')[0]], fields) for line in f]
            with codecs.open(os.path.join(path, 'captions.fear'),'r','utf8') as f:
                examples += [
                    data.Example.fromlist([line.split(',',1)[1], 'fear', line.split(',')[0]], fields) for line in f]
            with codecs.open(os.path.join(path, 'captions.sadness'),'r','utf8') as f:
                examples += [
                    data.Example.fromlist([line.split(',',1)[1], 'sadness', line.split(',')[0]], fields) for line in f]
        super(ED, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, id_field, split, shuffle=True ,root='.',path="../../datasets/EmotionDataset/lstm_data/noisy/", **kwargs):
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

        # LOAD TEST SAMPLES
        split_examples = cls(text_field, label_field, id_field, path="../../datasets/EmotionDataset/lstm_data/" + split + "/", **kwargs).examples
        random.shuffle(split_examples)
        dev_examples = []

        # LOAD TRAIN VOCAB SINCE I NEED IT TO RUN THE MODEL
        fields = [('text', text_field), ('label', label_field) , ('id', id_field)]
        train_examples = []
        train_path = "../../datasets/EmotionDataset/lstm_data/noisy"
        with codecs.open(os.path.join(train_path, 'captions.amusement'), 'r', 'utf8') as f:
            train_examples += [
                data.Example.fromlist([line.split(',',1)[1], 'amusement', line.split(',')[0]], fields) for line in f]
        with codecs.open(os.path.join(train_path, 'captions.anger'), 'r', 'utf8') as f:
            train_examples += [
                data.Example.fromlist([line.split(',',1)[1], 'anger', line.split(',')[0]], fields) for line in f]
        with codecs.open(os.path.join(train_path, 'captions.awe'), 'r', 'utf8') as f:
            train_examples += [
                data.Example.fromlist([line.split(',',1)[1], 'awe', line.split(',')[0]], fields) for line in f]
        with codecs.open(os.path.join(train_path, 'captions.contentment'), 'r', 'utf8') as f:
            train_examples += [
                data.Example.fromlist([line.split(',',1)[1], 'contentment', line.split(',')[0]], fields) for line in f]
        with codecs.open(os.path.join(train_path, 'captions.disgust'), 'r', 'utf8') as f:
            train_examples += [
                data.Example.fromlist([line.split(',',1)[1], 'disgust', line.split(',')[0]], fields) for line in f]
        with codecs.open(os.path.join(train_path, 'captions.excitement'), 'r', 'utf8') as f:
            train_examples += [
                data.Example.fromlist([line.split(',',1)[1], 'excitement', line.split(',')[0]], fields) for line in f]
        with codecs.open(os.path.join(train_path, 'captions.fear'), 'r', 'utf8') as f:
            train_examples += [
                data.Example.fromlist([line.split(',',1)[1], 'fear', line.split(',')[0]], fields) for line in f]
        with codecs.open(os.path.join(train_path, 'captions.sadness'), 'r', 'utf8') as f:
            train_examples += [
                data.Example.fromlist([line.split(',', 1)[1], 'sadness', line.split(',')[0]], fields) for line in f]

        print('train (for vocab initialization):',len(train_examples))
        print('split samples:',len(split_examples))

        return cls(text_field, label_field, id_field, examples=train_examples), cls(text_field, label_field, id_field, examples=train_examples), cls(text_field, label_field, id_field, examples=split_examples)

# load ED dataset
def load_ed(text_field, label_field, id_field, batch_size, split):
    print('loading data')
    train_data, aux_data, split_data = ED.splits(text_field, label_field, id_field, split)
    text_field.build_vocab(train_data, aux_data)
    label_field.build_vocab(train_data, aux_data)
    id_field.build_vocab(train_data, aux_data)

    # print('building batches')
    # train_iter, dev_iter = data.Iterator.splits(
    #     (train_data, dev_data), batch_sizes=(batch_size, len(dev_data)),repeat=False, shuffle=False,
    #     device = -1

    print('building batches')
    split_iter, aux_iter = data.Iterator.splits((split_data, aux_data), batch_sizes=(batch_size, batch_size, batch_size),repeat=False, shuffle=False,device = -1)
    print('Data built')
    return split_iter
#
# text_field = data.Field(lower=True)
# label_field = data.Field(sequential=False)
# train_iter, dev_iter , test_iter = load_mr(text_field, label_field, batch_size=50)
