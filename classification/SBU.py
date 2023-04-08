import os
import os.path
import numpy as np
import random
import torch.utils.data as data
from PIL import Image
import torch
from util import cal_subitizing
from torchvision import transforms
import json
NO_LABEL = -1


def make_union_dataset(root, edge=False):
    img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'images')) if f.endswith('.jpg')]
    label_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'labels')) if f.endswith('.png')]
    data_list = []
    if edge:
        edge_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'EdgeMasks')) if f.endswith('.png')]
        for img_name in img_list:
            if img_name in label_list:
                data_list.append((os.path.join(root, 'images', img_name + '.jpg'),
                                  os.path.join(root, 'labels', img_name + '.png'),
                                 os.path.join(root, 'EdgeMasks', img_name + '.png')))
            else:
                data_list.append((os.path.join(root, 'images', img_name + '.jpg'), -1, -1))
    else:
        for img_name in img_list:
            if img_name in label_list:
                data_list.append((os.path.join(root, 'images', img_name + '.jpg'),
                                  os.path.join(root, 'labels', img_name + '.png')))
            else:
                data_list.append((os.path.join(root, 'images', img_name + '.jpg'), -1))

    return data_list


def make_labeled_dataset(root, edge=False):
    img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'images')) if f.endswith('.jpg')]
    label_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'labels')) if f.endswith('.png')]
    data_list = []
    if edge:
        for img_name in label_list:
            data_list.append((os.path.join(root, 'images', img_name + '.jpg'),
                              os.path.join(root, 'labels', img_name + '.png'),
                              os.path.join(root, 'EdgeMasks', img_name + '.png')))
    else:
        for img_name in label_list:
            data_list.append((os.path.join(root, 'images', img_name + '.jpg'),
                                  os.path.join(root, 'labels', img_name + '.png')))
    return data_list


class SBU(data.Dataset):
    def __init__(self, root, subitizing=False, transform=None):
        self.root = root
        self.data = os.listdir(os.path.join(self.root, 'data'))
        self.label = os.listdir(os.path.join(self.root, 'label'))
        self.bound = os.listdir(os.path.join(self.root, 'bound'))
        self.transform = transform
        self.subitizing = subitizing

    def __getitem__(self, index):
        name = self.data[index]
        base_name = os.path.splitext(name)[0]
        img_path = os.path.join(self.root, 'data', name)
        gt_path = os.path.join(self.root, 'label', name)
        bound_path = os.path.join(self.root, 'bound', name)
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        bound = Image.open(bound_path).convert('L')

        if self.transform is not None:
            seed = np.random.randint(2147483647)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            
            img = self.transform(img)
            target = self.transform(target)
            bound = self.transform(bound)
            bound = (bound * 255).type(torch.int64)
        else:
            T = transforms.Compose([
                transforms.ToTensor()
            ])
            img = T(img)
            target = T(target)
            bound = T(bound)
            bound = (bound * 255).type(torch.int64)
        if self.subitizing:
            info = json.load(open(os.path.join(self.root, 'number.json')))
            number_per = info[self.data[index]]
            sample = {'image': img, 'label': target, 'bound': bound, 'number_per': number_per, 'name': base_name}
        else:
            sample = {'image': img, 'label': target, 'bound': bound, 'name': base_name}

        return sample

    def __len__(self):
        return len(self.data)


def relabel_dataset(dataset, edge_able=False):
    unlabeled_idxs = []
    for idx in range(len(dataset.imgs)):
        if not edge_able:
            path, label = dataset.imgs[idx]
        else:
            path, label, edge = dataset.imgs[idx]
        if label == -1:
            unlabeled_idxs.append(idx)
    labeled_idxs = sorted(set(range(len(dataset.imgs))) - set(unlabeled_idxs))

    return labeled_idxs, unlabeled_idxs




class MyDataSet(data.Dataset):


    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])

        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

def read_split_data(root: str, test_rate: float = 0.2):

    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)


    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()

    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for cla in flower_class:
        cla_path = os.path.join(root, cla)

        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        images.sort()

        image_class = class_indices[cla]

        every_class_num.append(len(images))

        val_path = random.sample(images, k=int(len(images) * test_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the labeled dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for test.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."
    indices = np.arange(len(train_images_path))
    random.shuffle(indices)
    random_train_path=[]
    random_train_label=[]
    for i in range(indices.size):
        random_train_path.append(train_images_path[indices[i]])
        random_train_label.append(train_images_label[indices[i]])



    return random_train_path, random_train_label, val_images_path, val_images_label


def read_unlabeled(path, length):
    images = os.listdir(path)
    indices = np.arange(len(images))

    random.seed(0)
    random.shuffle(indices)
    random_images = []

    for i in indices:
        random_images.append(images[i])

    images_path = []
    labels = []
    for img in random_images:
        img_path = os.path.join(path, img)
        images_path.append(img_path)
        labels.append(7)
        if len(images_path) > length:
            break
    print("{} images were found in the unlabeled dataset.".format(len(images_path)))
    return images_path, labels