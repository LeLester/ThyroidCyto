import torch.nn as nn
import torch


def l2(pred, target, size_average=True):
    b = pred.shape[0]
    l = 0.0
    for i in range(0, b):
        for j in range(0, b):
            l = l + (pred[i, j]-target[i, j])**2
    return l / (b**2)


class l2(torch.nn.Module):
    def __init__(self, size_average=True):
        super(l2, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        return l2(pred, target, self.size_average)


def temp_acc(pred, target):
    size = pred.shape[0]
    acc = 0.0
    for i in range(0, size):
        for j in range(0, size):
            t_acc = 1-abs(pred[i, j]-target[i, j])/target[i, j]
            acc = acc + t_acc
    acc = acc/(pred.shape[0])**2
    return acc
