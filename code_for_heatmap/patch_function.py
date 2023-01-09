import openslide
import cv2
from PIL import Image
import os
import numpy as np
from numpy import ndarray
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2


def get_2x(data_name):
    file_path = r'D:\prp\thyroid_gland_classification\data_deal\2x'
    result_path = os.path.join(r'D:\prp\thyroid_gland_classification\data_deal\make_label', data_name)
    os.mkdir(result_path)
    for file in os.listdir(file_path):
        if file[:6] == data_name:
            img = Image.open(os.path.join(file_path, file))
            img.save(os.path.join(result_path, file))


def build_locate(data):
    locate_point_x = []
    locate_point_y = []
    for file in os.listdir(data):
        x = file.split("_")[-5]
        y = file.split("_")[-3]
        locate_point_x.append(x)
        locate_point_y.append(y)
    return locate_point_x, locate_point_y


def cut_20x(root, data):
    l_x, l_y = build_locate(data)
    slide = openslide.OpenSlide(root)
    save_dir = os.path.join('/home/jiangli/zjc/left/20x/data')
    # os.mkdir(save_dir)
    for j in range(len(l_x)):
        x = l_x[j]
        y = l_y[j]
        print(x)
        print(y)
        for m in range(10):
            for n in range(10):
                img = slide.read_region((int(x) + 224 * m, int(y) + 224 * n), 0, (224, 224)).convert('RGB')
                save_path = '{}_{}_{}_{}_{}.png'.format(data.split("/")[-1], x, y, m, n)
                # save_path = '{}_{}_{}_{}_{}.png'.format(data[len(data)-6:], x, y, m, n)
                img.save(os.path.join(save_dir, save_path))
    slide.close()


def make_label(file_dir, data):
    save_dir = '/home/jiangli/zjc/left/2x_npy_gt'
    l_x, l_y = build_locate(data)
    for i in range(len(l_x)):
        file_name = l_x[i] + "_" + l_y[i]
        label = np.zeros((10, 10))
        for m in range(10):
            for n in range(10):
                file_temp = data.split("/")[-1] + "_" + file_name + "_" + str(m) + "_" + str(n) + '.png'
                img = Image.open(os.path.join(file_dir, file_temp))
                temp = np.sum(np.array(img) / (224 * 224 * 225))
                label[n, m] = temp
        save_path = os.path.join(save_dir, data.split("/")[-1]+ '_x_' + l_x[i] + "_y_" + l_y[i] + ".npy")
        np.save(save_path, label)


def npz_png(f1, f2):
    file_path = f1
    save_path = f2
    for file in os.listdir(file_path):
        temp = np.load(os.path.join(file_path, file))
        img = Image.fromarray(temp * 255)
        save_name = file[:-4] + '.png.png'
        img = img.convert('RGB')
        img.save(os.path.join(save_path, save_name))


file_path = '/home/jiangli/zjc/left/20x_gt'
wsi_path = '/home/jiangli/zjc/wsi_left'
for f_dir in os.listdir(wsi_path):
    for fi in os.listdir(os.path.join(wsi_path, f_dir)):
        print(fi)
        # get_2x(file[:-4])
        # file_name = file + '.svs'
        wsi = os.path.join(wsi_path, f_dir, fi)
        make = os.path.join('/home/jiangli/zjc/left/2x', fi[:-4])
        mask = os.path.join(file_path, fi[:-4])
        make_label(mask, make)
