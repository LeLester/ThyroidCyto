import argparse
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import random
import numpy as np
import sys
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import CADataset
from network import *
from loss import *
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--max_epochs', type=int, default=2500, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.0000001, help='segmentation network learning rate')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
args = parser.parse_args()

number = 0
logging_file = 'log/resnet_Log_{}.txt'.format(number)


def train(args, model):
    best_acc = 1
    logging.basicConfig(filename=logging_file, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    batch_size = args.batch_size
    T = transforms.Compose([transforms.RandomOrder([transforms.RandomHorizontalFlip(),
                                                    transforms.RandomVerticalFlip(),
                                                    transforms.RandomChoice([transforms.RandomRotation((0, 0)),
                                                                             transforms.RandomRotation((90, 90)),
                                                                             transforms.RandomRotation((180, 180)),
                                                                             transforms.RandomRotation((270, 270))])]),
                            transforms.ToTensor()
                            ])
    test_T = transforms.Compose([transforms.ToTensor()])

    trainset = CADataset('/home/jiangli/zjc/2x_heatmap/dataset/train', transform=T)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = CADataset('/home/jiangli/zjc/2x_heatmap/dataset/test', transform=test_T)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    criterion = nn.MSELoss()
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch, name = sampled_batch
            image, label = image_batch.cuda(), label_batch.cuda()
            mout = model(image)

            loss = criterion(mout, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            logging.info('iteration %d: loss: %.5f' % (iter_num, loss.item()))

        model.eval()
        test_loss_list = []
        for i_batch, batch in tqdm(enumerate(testloader)):
            image, label, name = batch
            image, label = image.cuda(), label.cuda()

            with torch.no_grad():
                mout = model(image)
                test_loss = criterion(mout, label)
            test_loss_list.append(test_loss.item())

        print('test loss: {}'.format(np.mean(test_loss_list)))
        if np.mean(test_loss_list) < best_acc:
            best_acc = min(best_acc, np.mean(test_loss_list))
            save_model_path = os.path.join('weight', '2x_pre_{}.pth'.format(number))
            torch.save(model.state_dict(), save_model_path)
    save_model_path = os.path.join('weight', '2x_pre_{}_final.pth'.format(number))
    torch.save(model.state_dict(), save_model_path)
    print('best test loss: {}'.format(best_acc))
    logging.info("final model saved")


if __name__ == "__main__":
    net = resnet50().cuda()
    train(args, net)
