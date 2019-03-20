import os
import random
import codecs
from torchtext import data


class SynthMMHS(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, split=None, **kwargs):
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
                with codecs.open(os.path.join(path, 'anns_train.txt'),'r','utf8') as f:
                    examples += [
                        data.Example.fromlist([line.split(',')[2], line.split(',')[1]], fields) for line in f]
            if split == 'val':
                with codecs.open(os.path.join(path, 'anns_val.txt'), 'r', 'utf8') as f:
                    examples += [
                        data.Example.fromlist([line.split(',')[2], line.split(',')[1]], fields) for line in f]

            # for example in examples:
            #     print example.label

        super(SynthMMHS, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, shuffle=True ,root='.',path="../../../datasets/HateSPic/SynthMMHS/anns/", **kwargs):
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
        train_examples = cls(text_field, label_field, path=path, split='train', **kwargs).examples
        if shuffle: random.shuffle(train_examples)

        dev_examples = cls(text_field, label_field, path=path, split='val', **kwargs).examples
        if shuffle: random.shuffle(dev_examples)

        # dev_index = int(len(examples) - 0.05 * len(examples))
        # train_examples = examples[0:dev_index]
        # dev_examples = examples[dev_index:]
        # random.shuffle(train_examples)
        # random.shuffle(dev_examples)

        print('train:',len(train_examples),'dev:',len(dev_examples))
        return cls(text_field, label_field, examples=train_examples), cls(text_field, label_field, examples=dev_examples)

def load_SynthMMHS(text_field, label_field, batch_size):
    print('loading data')
    train_data, dev_data = SynthMMHS.splits(text_field, label_field)
    text_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)
    print('Size vocab: ' + str(len(text_field.vocab.itos)))

    print('building batches')
    train_iter, dev_iter = data.Iterator.splits(
        (train_data, dev_data), batch_sizes=(batch_size, batch_size),repeat=False, shuffle=False,
        device = -1
    )

    return train_iter, dev_iter