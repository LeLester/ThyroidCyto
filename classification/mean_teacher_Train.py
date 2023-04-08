import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import ramps
import losses
from util import AverageMeter
from SBU import read_split_data, read_unlabeled
from SBU import MyDataSet
import sys

from model import  swin_tiny_patch4_window7_224 as swin_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0]

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./histology_dataset', help='Name of Experiment')

parser.add_argument('--labeled_path', type=str, default='')



parser.add_argument('--unlabeled_path', type=str, default='')



parser.add_argument('--exp', type=int,  default=65, help='model_name')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=1e-5, help='maximum epoch number to train')
parser.add_argument('--lr_decay', type=float,  default=0.9, help='learning rate decay')
parser.add_argument('--deterministic', type=int,  default=0, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--ema_decay', type=float,  default=0.9999, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.01, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=7.0, help='consistency_rampup')
parser.add_argument('--scale', type=int,  default=512, help='batch size of 8 with resolution of 416*416 is exactly OK')
parser.add_argument('--repeat', type=int,  default=3, help='repeat')
args = parser.parse_args()



model_weight_path = ""

rate = 0.01

print("rate=", rate,"device = ", device)

tb_writer = SummaryWriter(log_dir='result/{:.1f}%()'.format(rate * 100))
root_path = "weight/weight({:.1f}%)()".format(100 * rate)


Train_images_path, Train_images_label, Test_images_path, Test_images_label = read_split_data(args.labeled_path, 0.2)
unlabel_size = len(Train_images_label) + len(Test_images_label)
unlabeled_images_path, unlabeled_image_label = read_unlabeled(args.unlabeled_path, rate * unlabel_size)


batch_size = args.batch_size
epoch = args.epoch
base_lr = args.base_lr
lr_decay = args.lr_decay
loss_record = 0

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
else:
    cudnn.benchmark = True




def get_current_consistency_weight(epoch, max_epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, max_epoch)


def update_ema_variables(model, ema_model, alpha):

    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def create_model(ema=False):

    net = swin_model(num_classes=6)

    if ema:
        for param in net.parameters():
            param.detach_()
    net = torch.nn.DataParallel(net, device_ids=device_ids)
    net = net.cuda(device=device_ids[0])

    return net
def add_noise(image, lam=0.3):

    RGB = torch.exp(image) / torch.sum(torch.exp(image), dim=1, keepdim=True).to(device)
    B, C, H, W = image.shape


    mean = torch.zeros((B, C, H, W)).to(device)
    std = torch.std(image.view(B, C, H * W), dim=2, keepdim=True).expand(B, C, H * W).view(B, C, H, W)
    noise = torch.normal(mean, std) * lam * RGB
    noise = torch.clamp(noise, -0.2, 0.2)

    ema_image = torch.clamp(image + (1 - image) * noise, 0, 1)
    return ema_image


def evaluate(model, testloader, test_loss_record = None, loss_func = None):
    model.eval()
    if loss_func == None:
        loss_func = torch.nn.CrossEntropyLoss()
    if test_loss_record == None:
        test_loss_record = AverageMeter()
    start = time.time()
    true =0
    num_sample = 0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            images, labels = data
            images = images.cuda(device)
            labels = labels.cuda(device)
            pred = model(images)
            loss = loss_func(pred, labels)

            num_sample += images.shape[0]
            try:
                true += torch.eq(torch.argmax(pred,dim=1),labels).sum()
            except :
                print("images.shape = ", images.shape[0])
                print("pred.shape = ", pred.shape[0])
                print("label.shape=",labels.shape[0])


            test_loss_record.update(loss.item(), batch_size, pred.detach().cpu(), labels.detach().cpu())
            end = time.time()
            cost = (end - start)
            progress = (i+1) / len(testloader)
            eta = int(cost / progress - cost)
            print("\r", end='')
            print('test dataset, progress: %.2f%% , loss: %.5f, acc: %.5f, eta: %d min %d sec' %
                  (100 * progress, test_loss_record.avg, test_loss_record.acc, eta/60, eta % 60), end=' ')
    print()
    return test_loss_record.acc


def train_label(model, ema_model, trainloader, train_loss_record, train_con_loss_record, epoch_num, max_epoch, T_noise):

    start = time.time()
    for i_batch, sampled_batch in enumerate(trainloader):

        image_batch, label_batch = sampled_batch
        image_batch = image_batch.cuda(device)
        label_batch = label_batch.cuda(device)


        ema_inputs = T_noise(image_batch)

        pred = model(image_batch)
        with torch.no_grad():
            ema_pred = ema_model(ema_inputs)

        Loss = loss_func(pred, label_batch)
        Con_loss = consistency_criterion(pred, ema_pred)

        supervised_loss = Loss
        consistency_loss = Con_loss
        consistency_weight = get_current_consistency_weight(epoch_num, max_epoch)

        loss = supervised_loss + consistency_weight * consistency_loss


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        update_ema_variables(model, ema_model, args.ema_decay)

        train_loss_record.update(Loss.item(), batch_size, pred.detach().cpu(), label_batch.detach().cpu())
        train_con_loss_record.update(Con_loss.item(), batch_size, ema_pred.detach().cpu(), label_batch.detach().cpu())
        end = time.time()
        cost = (end - start)
        progress = (i_batch + 1) / len(trainloader)
        eta = int(cost / progress - cost)
        print("\r", end='')
        print(
            'labeled dataset, progress: %.2f%% , epoch %d : loss: %.5f,  con loss: %.5f,  loss_weight: %.5f, lr: %.5f, teacher_acc: %.5f, student_acc: %.5f, eta: %d min %d sec' %
            (100 * progress, epoch_num, train_loss_record.avg, train_con_loss_record.avg, consistency_weight,
             optimizer.param_groups[1]['lr'], train_con_loss_record.acc, train_loss_record.acc, eta / 60, eta % 60),
            end=' ')
    print()
def train_unlabel(model, ema_model, unlabelloader, unlabeled_con_loss_record, consistency_criterion, epoch_num, max_epoch, T_noise):
    start = time.time()
    for i_batch, sampled_batch in enumerate(unlabelloader):
        image_batch, label_batch = sampled_batch
        image_batch = image_batch.cuda(device)

        ema_inputs = T_noise(image_batch)

        pred = model(image_batch)
        with torch.no_grad():
            ema_pred = ema_model(ema_inputs)

        Con_loss = consistency_criterion(pred, ema_pred)

        consistency_loss = Con_loss
        consistency_weight = get_current_consistency_weight(epoch_num, max_epoch)

        loss = consistency_weight * consistency_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        update_ema_variables(model, ema_model, args.ema_decay)
        unlabeled_con_loss_record.update(Con_loss.item(), batch_size)

        end = time.time()
        cost = (end - start)
        progress = (i_batch + 1) / len(unlabelloader)
        eta = int(cost / progress - cost)
        print("\r", end='')
        print('unlabeled dataset, progress: %.2f%% , epoch: %d  con loss: %.5f,  loss_weight: %.5f, eta: %d min %d sec' %
            (100 * progress, epoch_num, unlabeled_con_loss_record.avg,  consistency_weight,
            eta / 60, eta % 60), end=' ')
    print()


if __name__ == "__main__":

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    model = create_model()
    model.load_state_dict(torch.load(model_weight_path, map_location=device))


    ema_model = create_model(ema=True)
    ema_model.load_state_dict(torch.load(model_weight_path, map_location=device))

    T_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    T_noise = transforms.Compose([transforms.RandomResizedCrop(224),
                                  transforms.RandomVerticalFlip(),
                                  transforms.GaussianBlur(11, 5)
                                  ])
    T_test = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    db_train = MyDataSet(Train_images_path, Train_images_label, T_train)
    db_test = MyDataSet(Test_images_path, Test_images_label, T_test)
    db_unlabeled = MyDataSet(unlabeled_images_path, unlabeled_image_label, T_train)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=False,pin_memory=True,num_workers=nw)
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=nw)
    unlabelloader = DataLoader(db_unlabeled,batch_size=batch_size,shuffle=False,pin_memory=True,num_workers=nw)

    model.train()
    ema_model.train()

    optimizer = optim.SGD([
        {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * base_lr},
        {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias'],
         'lr': base_lr, 'weight_decay': 0.0005}
    ], momentum=0.9)


    if args.consistency_type == 'sig_mse':
        consistency_criterion = losses.sigmoid_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = F.kl_div
    elif args.consistency_type == 'mse':
        consistency_criterion = F.mse_loss
    else:
        assert False, args.consistency_type

    logging.info("{} itertations per epoch".format(len(trainloader)))
    loss_func = torch.nn.CrossEntropyLoss()

    lr_ = base_lr

    max_test_acc = evaluate(ema_model, testloader, None, loss_func)
    print()
    for epoch_num in range(epoch):
        train_loss_record, train_con_loss_record, unlabeled_con_loss_record, test_loss_record, test_con_loss_record \
            = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        train_unlabel(model, ema_model, unlabelloader, unlabeled_con_loss_record, consistency_criterion, epoch_num, epoch,T_noise)


        if rate<1:
            train_label(model, ema_model, trainloader, train_loss_record, train_con_loss_record, epoch_num, epoch, T_noise)
        else:
            for k in range(int(rate * 2)):
                print("k = ",k)
                train_label(model, ema_model, trainloader, train_loss_record, train_con_loss_record, epoch_num, epoch, T_noise)
                print()

        evaluate(ema_model, testloader, test_loss_record, loss_func)
        tag = ["train_loss", "labeled_con_loss", "unlabeled_con_loss", "test_loss", "train_student_acc", "train_teacher_acc", "test_acc"]
        tb_writer.add_scalar(tag[0], train_loss_record.avg, epoch_num)
        tb_writer.add_scalar(tag[1], train_con_loss_record.avg, epoch_num)
        tb_writer.add_scalar(tag[2], unlabeled_con_loss_record.avg, epoch_num)
        tb_writer.add_scalar(tag[3], test_loss_record.avg, epoch_num)
        tb_writer.add_scalar(tag[4], train_loss_record.acc, epoch_num)
        tb_writer.add_scalar(tag[5], train_con_loss_record.acc,epoch_num)
        tb_writer.add_scalar(tag[6], test_loss_record.acc, epoch_num)

        if test_loss_record.acc > max_test_acc:
            max_test_acc = test_loss_record.acc
            save_mode_path = root_path + "/teacher-model{}({})".format(epoch_num, max_test_acc)
            if not os.path.exists(root_path):
                os.mkdir(root_path)
            torch.save(ema_model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
        print()

