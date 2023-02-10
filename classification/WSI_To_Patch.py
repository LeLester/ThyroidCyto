import openslide
import os
import numpy as np

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
    l_x, l_y ,block_names= build_locate(data)
    slide = openslide.OpenSlide(root)
    save_dir = os.path.join('/home/jiangli/yx/thyroid/WSI-full-patch', data.split("/")[-1])
    if os.path.exists(save_dir)==False:
        os.mkdir(save_dir)

    threshold = get_threshold(data)
    for j in range(len(l_x)):
        print("\r", end="")
        print("progress: {}%: ".format(100 * (j+1) / len(l_x)),  end="")
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


wsi_path = ''
file_path = ''

for file in os.listdir(wsi_path):

    print(file)
    wsi = os.path.join(wsi_path, file)
    make = os.path.join(file_path, file[:-4])
    cut_20x(wsi, make)

    print()


