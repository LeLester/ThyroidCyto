import os
import json
import math
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from utils import GradCAM, show_cam_on_image, center_crop_img, read_split_data_5folds
from model import swin_large_patch4_window7_224_in22k as create_model


class ResizeTransform:
    def __init__(self, im_h: int, im_w: int):
        self.height = self.feature_size(im_h)
        self.width = self.feature_size(im_w)

    @staticmethod
    def feature_size(s):
        s = math.ceil(s / 4)
        s = math.ceil(s / 2)
        s = math.ceil(s / 2)
        s = math.ceil(s / 2)
        return s

    def __call__(self, x):
        result = x.reshape(x.size(0),
                           self.height,
                           self.width,
                           x.size(2))


        result = result.permute(0, 3, 1, 2)

        return result


def main(img_path, target_class, target):
    img_size = 224
    data_transform = transforms.Compose(
         [transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)
    model.eval()
    img_size = 224
    assert img_size % 32 == 0
    target_layers = [model.norm]
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False,
                  reshape_transform=ResizeTransform(im_h=img_size, im_w=img_size))
    target_category = target-1
    grayscale_cam = cam(input_tensor=img.to(device), target_category=target_category)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img = center_crop_img(img, img_size)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img / 255., grayscale_cam, use_rgb=True)
    print(visualization.shape)
    im = Image.fromarray(visualization)
    im.save("dataset/CAM/"+str(target)+"/"+img_path.split('/')[-1])


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=6).to(device)
    model_weight_path = "weights/model-133.pth"
    _, _, test_images_path, test_images_label = read_split_data_5folds("dataset/dataset_new/",0.2)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    for i in range(len(test_images_path)):
        print("\r", end="")
        print("Progress: {}%: ".format(100 * i/ test_images_path.__len__()), end="")
        main(test_images_path[i],  model, test_images_label[i]+1)


