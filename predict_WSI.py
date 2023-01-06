
import os
import json
import math
import sys
import time

import numpy
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from model import swin_large_patch4_window7_224_in22k as create_model
import openslide


def main(imgs,model):

    img_size = 224
    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    img_path = dir_file_path
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)


    imgs = [data_transform(img) for img in imgs]
    imgs = torch.stack(imgs)

    with torch.no_grad():
        output = model(imgs.to(device)).cpu()
        predict = torch.softmax(output, dim=1)
        predict_cla = torch.argmax(predict,1).numpy()
        imgs.cpu()
        torch.cuda.empty_cache()


    return predict_cla


color = [
    [153, 102, 255],  # 紫色，I
    [255, 255, 0],  # 黄色。Ⅱ
    [102, 204, 0],  # 绿色，Ⅲ
    [0, 255, 255],  # 蓝色，Ⅳ
    [255, 0, 255],  # 粉色，Ⅴ
    [255, 0, 0], # 大红，Ⅵ
    [255, 255, 255] # 白色背景
]
Slide_path = "/home/jiangli/yx/thyroid/WSI/1"

if __name__ == '__main__':


    WSIs = os.listdir("/home/jiangli/yx/thyroid/WSI-select-patch/1")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


    model = create_model(num_classes=6).to(device)

    model_weight_path = "weights/model-133.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    start = time.time()

    for WSI in WSIs:


        WSI_path = os.path.join("/home/jiangli/yx/thyroid/WSI-select-patch/1", WSI)

        slide = openslide.OpenSlide(Slide_path + "/" + WSI + ".svs")
        X = slide.level_dimensions[0][0]
        Y = slide.level_dimensions[0][1]
        print("X = ", X, " Y=", Y)


        visualization = 150 * np.ones((X//224, Y//224, 3),dtype=np.uint8)
        print("visualization.shape = ", visualization.shape)


        pics = os.listdir(WSI_path)
        print(WSI, "的patch数为：", pics.__len__())

        c = [0, 0, 0, 0, 0, 0]

        i=0
        imgs = []

        batchsize = 1



        for pic in pics:

            dir_file_path = os.path.join(WSI_path, pic)
            img = Image.open(dir_file_path)
            imgs.append(img)
            i +=1

            if i % batchsize== 0:

                ress = main(imgs, model)

                print("\r", end="")
                print("进度: {}%: ".format(100*i / pics.__len__()), end="")
                sys.stdout.flush()


                for j in range(batchsize):
                    c[ress[j]] += 1

                    x = int(pic.split("_")[1]) // 224 + int(pic.split("_")[3])
                    y = int(pic.split("_")[2]) // 224 + int(pic.split("_")[4][0])
                    visualization[x][y] = color[ress[j]]
                imgs = []


            if i % batchsize != 0 and i == pics.__len__():
                ress = main(imgs, model)
                for j in range(len(ress)):
                    c[ress[j]] += 1

                    x = int(pic.split("_")[1]) // 224 + int(pic.split("_")[3])
                    y = int(pic.split("_")[2]) // 224 + int(pic.split("_")[4][0])
                    visualization[x][y] = color[ress[j]]
                imgs = []

        visualization = Image.fromarray(visualization)
        visualization.save("/home/jiangli/yx/thyroid/WSI-Visualization/1/" + WSI + ".png")

        print("WSI：", WSI, "在总共 {} patches中".format(numpy.array(c).sum())," , 预测结果为：")
        for i in range(6):
            print("第{}类个数".format(i+1), c[i])
    end = time.time()
    print("总耗时：",end-start)

