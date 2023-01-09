import os
import numpy as np
import matplotlib.pyplot as plt


def make_map(p1, sa_name):
    temp = np.load(p1)
    plt.imshow(temp, cmap='Reds')
    plt.colorbar()
    sa_name = sa_name+'.png'
    # save_name = os.path.join(p2, sa_name)
    save_name = os.path.join(sa_name)
    # plt.show()
    plt.savefig(save_name)


file_path = "/home/jiangli/zjc/left/2x_pred"
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
    # score = []
    for file in os.listdir(os.path.join(file_path, file_dir)):
        m = int(int(file.split("_")[-3])/2240)
        n = int(int(file.split("_")[-1][:-4])/2240)
        temp = np.load(os.path.join(file_path, file_dir, file))
        s = np.sum(temp)
        result[n, m] = s
        # print(m,n)
        # print(s)
    file_save = '/home/jiangli/zjc/left/npy_pred'
    save_name = file_dir + '.npy'
    print(save_name)

    np.save(os.path.join(file_save, save_name), result)
