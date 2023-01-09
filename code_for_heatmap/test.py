import logging
import os
import random
import sys
import numpy as np
from loss import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import CADataset
from network import *


batch_size = 1
img_size = 224
number = 0
test_save_dir = 'prediction_{}'.format(number)
logging_file = "log/CA25NetTestLog_{}.txt".format(number)


def inference(model, test, img_num, test_save_dir=None):
    testset = CADataset(test)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    if test_save_dir is not None:
        logging.info("prediction saved to {}".format(test_save_dir))
        if not os.path.exists('/home/jiangli/zjc/left/2x_pred/{}'.format(img_num)):
            os.mkdir('/home/jiangli/zjc/left/2x_pred/{}'.format(img_num))
    else:
        logging.info("prediction results not saved")
    model.eval()
    acc_list = []
    for i_batch, batch in tqdm(enumerate(testloader)):
        # image, label, bound, name = batch
        image, name = batch
        image = image.cuda()

        with torch.no_grad():
            mout = model(image)
            mout = mout.squeeze(0).detach().cpu().numpy()

        if test_save_dir is not None:
            mout = np.squeeze(np.array(mout))
            save_path = os.path.join('/home/jiangli/zjc/left/2x_pred/{}'.format(img_num), name[0][:-4])
            save_path = save_path + '.npy'
            np.save(save_path, mout)
    print("Testing Finished!")


if __name__ == "__main__":

    net = resnet50().cuda()

    model_path = './weight/2x_pre_{}.pth'.format(number)
    net.load_state_dict(torch.load(model_path))

    file_path = '/home/jiangli/zjc/left/2x'
    for file in os.listdir(file_path):
        test = os.path.join(file_path, file)
        num = file
        inference(net, test, file, test_save_dir)

