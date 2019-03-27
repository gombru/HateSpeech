import os
import random
import codecs
from torchtext import data


class MMHS50K(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, split=None, **kwargs):

        # text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]
        if examples is None:
            path = self.dirname if path is None else path
            examples = []

            if split == 'train':
                with codecs.open(os.path.join(path, 'tweets.train'),'r','utf8') as f:
                    examples += [
                        data.Example.fromlist([line.split(',')[1], float(line.split(',')[2])], fields) for line in f]

            if split == 'val':
                with codecs.open(os.path.join(path, 'tweets.val'), 'r', 'utf8') as f:
                    examples += [
                        data.Example.fromlist([line.split(',')[1], float(line.split(',')[2])], fields) for line in f]

        super(MMHS50K, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, shuffle=True ,root='.',path="../../../datasets/HateSPic/MMHS/lstm_data/lstm_data_50k_3workers_regression/", **kwargs):

        train_examples = cls(text_field, label_field, path=path, split='train', **kwargs).examples
        if shuffle: random.shuffle(train_examples)

        dev_examples = cls(text_field, label_field, path=path, split='val', **kwargs).examples
        if shuffle: random.shuffle(dev_examples)

        print('train:',len(train_examples),'dev:',len(dev_examples))
        return cls(text_field, label_field, examples=train_examples), cls(text_field, label_field, examples=dev_examples)

def load_MMHS50K(text_field, label_field, batch_size):
    print('loading data')
    train_data, dev_data = MMHS50K.splits(text_field, label_field)
    text_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)
    print('Size vocab: ' + str(len(text_field.vocab.itos)))

    print('building batches')
    train_iter, dev_iter = data.Iterator.splits(
        (train_data, dev_data), batch_sizes=(batch_size, batch_size),repeat=False, shuffle=False,
        device = -1
    )

    return train_iter, dev_iter