import openslide
import cv2
from PIL import Image
import os
import numpy as np
from tqdm import tqdm


root = '/home/jiangli/zjc/wsi_left'
d = os.listdir(root)
for i in d:
    p1 = os.path.join(root, i)
    WSIs = os.listdir(p1)
    for wsi_name in tqdm(WSIs):
        print(wsi_name)
        base_name = os.path.splitext(wsi_name)[0]
        wsi_path = os.path.join(p1, wsi_name)
        slide = openslide.OpenSlide(wsi_path)
        print(slide.level_dimensions)
        print(len(slide.level_dimensions))
        if len(slide.level_dimensions) != 1:
            print(base_name)
            width, height = slide.level_dimensions[1]
            Nwidth, Nheight = int(width / 560), int(height / 560)
            save_dir = os.path.join('/home/jiangli/zjc/left/2x', base_name)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            for i in range(Nwidth):
                for j in range(Nheight):
                    x, y = i * 2240, j * 2240
                    img = slide.read_region((x, y), 1, (560, 560)).convert('RGB')
                    img = img.resize((224, 224))
                    save_name = '{}.png'.format('{}_x_{}_y_{}_{}_{}.png'.format(base_name, x, y, i, j))
                    img.save(os.path.join(save_dir, save_name))
        slide.close()
