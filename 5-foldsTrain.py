# 实现了五折交叉验证

import os
import argparse

import numpy
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet

from model import  swin_large_patch4_window7_224_in22k as create_model
from utils import read_split_data_5folds, train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    Train_images_path, Train_images_label, Test_images_path, Test_images_label = read_split_data_5folds(args.data_path,0.2)

    img_size = 224

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test": transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }


    Test_dataset = MyDataSet(images_path=Test_images_path,
                            images_class=Test_images_label,
                            transform=data_transform["test"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))


    model = create_model(num_classes=args.num_classes).to(device)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)

    for epoch in range(args.epochs):
        val_images_path = []
        val_images_label = []
        train_images_path = []
        train_images_label = []
        k = epoch % 5

        for i in range(5):
            length = int(Train_images_path.__len__()/5)
            if i == k:
                val_images_path = np.concatenate((val_images_path, Train_images_path[i*length: (i+1)*length]))
                val_images_label = np.concatenate((val_images_label, Train_images_label[i*length: (i+1)*length]))

                if i==4 and (i+1)*length!=int(Train_images_path.__len__()):
                    val_images_path = np.concatenate((val_images_path, Train_images_path[(i + 1) * length:]))
                    val_images_label = np.concatenate((val_images_label, Train_images_label[(i + 1) * length:]))
            else:
                train_images_path = np.concatenate((train_images_path, Train_images_path[i * length: (i + 1) * length]))
                train_images_label = np.concatenate((train_images_label, Train_images_label[i * length: (i + 1) * length]))
                if i==4 and (i+1)*length!=int(Train_images_path.__len__()):
                    train_images_path = np.concatenate((train_images_path, Train_images_path[(i + 1) * length:]))
                    train_images_label = np.concatenate((train_images_label, Train_images_label[(i + 1) * length:]))

        val_images_label = val_images_label.astype(int)
        train_images_label = train_images_label.astype(int)


        train_dataset = MyDataSet(images_path=train_images_path,
                                  images_class=train_images_label,
                                  transform=data_transform["train"])


        val_dataset = MyDataSet(images_path=val_images_path,
                                 images_class=val_images_label,
                                 transform=data_transform["val"])

        print(type(train_dataset.images_path))
        print("model{} :".format(k+1), "{} images for training".format(len(train_images_path)), "{} images for val".format(len(val_images_path)),"{} images for test".format(len(Test_images_path)))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=nw, collate_fn=train_dataset.collate_fn)

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=nw, collate_fn=val_dataset.collate_fn)

        test_loader = torch.utils.data.DataLoader(Test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=nw, collate_fn=Test_dataset.collate_fn)
        # train
        train_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader, device=device, epoch=epoch)

        # validate
        val_loss, val_acc = evaluate(model=model, data_loader=val_loader, device=device, epoch=epoch)

        test_loss, test_acc = evaluate(model=model, data_loader=test_loader, device=device, epoch=epoch, tag=1)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate", "test_loss", "test_acc" ]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[5], test_loss, epoch)
        tb_writer.add_scalar(tags[6], test_acc, epoch)

        torch.save(model.state_dict(), "weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.00001)


    parser.add_argument('--data-path', type=str,
                        default="dataset/dataset_new")

    parser.add_argument('--weights', type=str, default='',help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
