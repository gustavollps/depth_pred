# import fix
# fix.fix()

import cv2
import pickle
import torch
import numpy as np
from datasets.nyudepthv2 import NYUDepthV2
from datasets.make3d import Make3D
from cnn.resnet_new import resnet50
from cnn.inception_resnet_v2 import InceptionResNetV2
from torch.utils.data import DataLoader
import os
import losses
from draw import draw_debug
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dataset = pickle.load(open('dataset_list.pickle', 'rb'))
dataset_temp = Make3D(test=True)


def getDataLoader(batch_size, imgs_obj):
    return DataLoader(imgs_obj, batch_size=batch_size, shuffle=False)


def testNet(net, debug=False):
    net.eval()
    dataset_temp = Make3D(test=True)
    dataset = getDataLoader(1, dataset_temp)
    losses_list = [losses.RelLoss()]  # , losses.RMSELoss(), losses.Log10Loss()]
    total_loss = [0, 0, 0]
    print("Testing......")
    if debug:
        print("Test set size: {:d}".format(len(dataset_temp)))
    for index, loss in enumerate(losses_list, 0):
        for i, data in enumerate(dataset, 0):
            inputs, labels, mask = data
            inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
            outputs = net(inputs.float())
            loss_size = loss(outputs, labels, mask)
            total_loss[index] = total_loss[index] + loss_size.data

            if debug:
                if (i + 1) % (len(dataset) // 3 + 1) == 0:
                    print("{:d}%".format(int(i / len(dataset) * 100)))
                if (i + 1) % (len(dataset) // 10 + 1) == 0:
                    draw_debug(inputs, labels, mask, outputs, name='test')

        total_loss[index] = total_loss[index] / len(dataset)

    print('Test results:\n\trel: {:.6f}'.format(total_loss[0]))
    return total_loss[0]


if __name__ == '__main__':
    net = InceptionResNetV2().to(device)
    net, _, _, _, _ = utils.restore_checkpoint(net, None, None, inception=True, masked=True, recent=False)
    net.eval()
    print(testNet(net))
