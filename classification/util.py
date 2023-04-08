
import os
import pickle

import numpy as np
import torch
from torch.utils.data.sampler import Sampler

from skimage import measure, color, morphology
import itertools
from PIL import Image



def load_model(path):

    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)


        N = checkpoint['state_dict']['top_layer.bias'].size()


        sob = 'sobel.0.weight' in checkpoint['state_dict'].keys()
        model = models.__dict__[checkpoint['arch']](sobel=sob, out=int(N[0]))


        def rename_key(key):
            if not 'module' in key:
                return key
            return ''.join(key.split('.module'))

        checkpoint['state_dict'] = {rename_key(key): val
                                    for key, val
                                    in checkpoint['state_dict'].items()}


        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded")
    else:
        model = None
        print("=> no checkpoint found at '{}'".format(path))
    return model


class UnifLabelSampler(Sampler):


    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        size_per_pseudolabel = int(self.N / len(self.images_lists)) + 1
        res = np.zeros(size_per_pseudolabel * len(self.images_lists))

        for i in range(len(self.images_lists)):
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res[i * size_per_pseudolabel: (i + 1) * size_per_pseudolabel] = indexes

        np.random.shuffle(res)
        return res[:self.N].astype('int')

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return self.N

class TwoStreamBatchSampler(Sampler):

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):

    args = [iter(iterable)] * n
    return zip(*args)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.acc = 0
        self.cnt = 0


    def update(self, val, n=1, pred = None, label = None):

        if pred != None and label != None:
            pred = np.argmax(np.array(pred),axis=1)
            for i in range(len(label)):
                if(pred[i] == label[i]):
                    self.cnt += 1

            self.val = val
            self.sum += val * len(label)
            self.count += len(label)
            self.avg = self.sum / self.count
            self.acc = self.cnt / self.count
        else:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr


class Logger():


    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)



def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cal_subitizing(label, threshold=6, min_size_per=0.005):
    label = np.array(label.convert('1'))

    dst = morphology.remove_small_objects(label, min_size=min_size_per*label.shape[0]*label.shape[1], connectivity=2)#remove small connected areas with a threshold
    labels = measure.label(dst, connectivity=2, background=0)
    number = labels.max()+1
    number = min(number, threshold)

    number_per = number

    percentage = np.sum(label)/(label.shape[0]*label.shape[1])
    return number_per, percentage

def relabel_dataset(dataset):
    unlabeled_idxs = []
    for idx in range(len(dataset.imgs)):
        if dataset.imgs[idx][1] == -1:
            unlabeled_idxs.append(idx)
    labeled_idxs = sorted(set(range(len(dataset.imgs))) - set(unlabeled_idxs))

    return labeled_idxs, unlabeled_idxs


