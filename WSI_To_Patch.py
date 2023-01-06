import sys
import time

import openslide
import cv2
from PIL import Image
import os
import numpy as np
from numpy import ndarray
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import predict_WSI

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
    block_names = []
    for file in os.listdir(data):
        x = file.split("_")[2]
        y = file.split("_")[4][:-4]
        locate_point_x.append(x)
        locate_point_y.append(y)
        block_names.append(file)
    return locate_point_x, locate_point_y,block_names


def get_threshold(block_label):
    block_names = os.listdir(block_label)
    nums = np.array([])
    all = 100 * len(block_names)
    for block in block_names:
        matrix = np.load(os.path.join(block_label, block), allow_pickle=True)
        for i in range(10):
            for j in range(10):
                x = matrix[i][j]
                nums = np.append(nums, x)
    nums.sort()

    return nums[int(all * 0.90)]




def cut_20x(root, data):
    # 得到所有Block的坐标
    l_x, l_y ,block_names= build_locate(data)
    slide = openslide.OpenSlide(root)

    save_dir = os.path.join('/home/jiangli/yx/thyroid/WSI-select-patch/2', data.split("/")[-1])
    if os.path.exists(save_dir)==False:
        os.mkdir(save_dir)


    threshold = get_threshold(data)



    for j in range(len(l_x)):
        print("\r", end="")
        print("进度: {}%: ".format(100 * (j+1) / len(l_x)),  end="")



        x = l_x[j]
        y = l_y[j]
        block_name = block_names[j]

        matrix = np.load(data+'/'+block_name)

        for m in range(10):
            for n in range(10):
                if matrix[m][n] <= threshold:
                    continue

                img = slide.read_region((int(x) + 224 * m, int(y) + 224 * n), 0, (224, 224)).convert('RGB')

                save_path = '{}_{}_{}_{}_{}.png'.format(data.split("/")[-1], x, y, m, n)
                img.save(os.path.join(save_dir, save_path))
    slide.close()



def make_label(file_dir, data):
    save_dir = '/home/jiangli/zjc/data_deal/2x_npy'
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



wsi_path = '/home/jiangli/yx/thyroid/WSI/2'
file_path = '/home/jiangli/yx/thyroid/2x_label_dir'


for file in os.listdir(wsi_path):
    print(file)
    wsi = os.path.join(wsi_path, file) # 源地址
    make = os.path.join(file_path, file[:-4]) # 目标地址

    cut_20x(wsi, make)
    print()


