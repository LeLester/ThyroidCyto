import os
import numpy as np
import matplotlib.pyplot as plt

# draw the heatmap
def make_map(p1, sa_name):
    temp = np.load(p1)
    plt.imshow(temp, cmap='Reds')
    plt.colorbar()
    sa_name = sa_name+'.png'
    save_name = os.path.join(sa_name)
    plt.savefig(save_name)


"""
function to convert 2x predicted result to wsi predict result
file_path: 2x predicted result 
save_path: wsi predict result
"""

def make_wsi_npy(file_path, save_path):
    for file_dir in os.listdir(file_path):
        m_list = []
        n_list = []
        for file in os.listdir(os.path.join(file_path, file_dir)):
            m = int(int(file.split("_")[-3]) / 2240)
            n = int(int(file.split("_")[-1][:-4]) / 2240)
            m_list.append(m)
            n_list.append(n)
        m_max = np.max(np.array(m_list))
        n_max = np.max(np.array(n_list))
        result = np.zeros((n_max+1, m_max+1))
        for file in os.listdir(os.path.join(file_path, file_dir)):
            m = int(int(file.split("_")[-3])/2240)
            n = int(int(file.split("_")[-1][:-4])/2240)
            temp = np.load(os.path.join(file_path, file_dir, file))
            s = np.sum(temp)
            result[n, m] = s
        save_name = file_dir + '.npy'
        print(save_name)
        np.save(os.path.join(file_save, save_name), result)
