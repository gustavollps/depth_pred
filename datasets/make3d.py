import scipy.io
import os
import cv2
import numpy as np
import numpy.ma as ma
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Lambda, Normalize, ToTensor
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import PIL
from draw import draw_tensor
from transforms import *


class Make3D(Dataset):
    def __init__(self, test=False, masked=True, data_augmentation=1):
        # depth['Position3DGrid'].max()
        self.max_value = 81.92119622492302
        self.masked = masked
        self.variations = data_augmentation

        if test is True:
            self.depth_path = '/home/gustavo/pytorch/Make3D/Test134Depth'
            self.image_path = '/home/gustavo/pytorch/Make3D/Test134Img'
        else:
            self.depth_path = '/home/gustavo/pytorch/Make3D/Train400Depth'
            self.image_path = '/home/gustavo/pytorch/Make3D/Train400Img'
        self.test = test

        # depth_files = [os.path.join(file_path, o) for o in os.listdir(file_path)]
        self.depth_files = [o for o in os.listdir(self.depth_path)]
        # self.image_files = [o for o in os.listdir(image_path)]
        # self.resize_factor = 7
        self.test_size = (192, 256)  # (345, 460)
        self.train_size = (192, 256)  # (170, 223)
        # self.sample_size = (int(1704 / self.resize_factor), int(2272 / self.resize_factor))

        self.trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
        self.trans_depth = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        self.trans_mask = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

    def __getitem__(self, index_full):
        index = index_full // self.variations
        variation = index_full % self.variations

        if self.test is True:
            size = self.test_size
        else:
            size = self.train_size

        file = self.depth_files[index]
        depth = scipy.io.loadmat(os.path.join(self.depth_path, file))
        image_file = 'img' + file[file.index('-'):len(file) - 3] + 'jpg'
        image = cv2.resize(cv2.imread(os.path.join(self.image_path, image_file)), size)

        depth_prep = np.swapaxes(depth['Position3DGrid'], 0, 2)[3]
        depth_prep = np.transpose(depth_prep) / self.max_value * 255
        depth_prep = cv2.resize(depth_prep, size, cv2.INTER_NEAREST)
        depth_prep = depth_prep.astype('uint8')

        kernel = np.ones((6, 6), np.uint8)

        if self.test:
            output_depth = self.trans_depth(depth_prep)
            output_img = self.trans(image)
            mask = np.transpose(np.swapaxes(self.get_mask(output_depth, output_img, self.masked), 0, 2))
            mask = cv2.erode(mask, kernel)
            return output_img, output_depth, mask

        else:
            image = np.swapaxes(image, 0, 2)
            depth_prep = np.swapaxes(np.expand_dims(depth_prep, axis=2), 0, 2)
            merged = np.swapaxes(np.concatenate([image, depth_prep], axis=0), 1, 2)
            output_img = random_transform(torch.from_numpy(merged), (size[1], size[0]))
            # output_depth = sequential_transform(variation, size, torch.from_numpy(depth_prep))
            output_depth = output_img[3].unsqueeze(0)
            output_img = output_img[0:3]
            mask = np.transpose(np.swapaxes(self.get_mask(output_depth, output_img, self.masked), 0, 2))
            mask = cv2.erode(mask, kernel)
            return output_img, output_depth, mask

    def __len__(self):
        if self.test:
            return len(self.depth_files)
        else:
            # return 10
            return len(self.depth_files) * self.variations

    def get_mask(self, dep, img, masked):
        img_dep = dep.detach().numpy()
        if masked:
            img_mask_1 = ma.masked_not_equal(img_dep, 0)
            img_mask_2 = ma.masked_less(img_dep, (70 / self.max_value))
            img_mask = ~img_mask_1.mask + ~img_mask_2.mask
            img_mask = np.array(img_mask, dtype=np.uint8)
            return 1 - img_mask
        else:
            mask = np.all(np.swapaxes(img.detach().numpy(), 0, 2) != [0, 0, 0], axis=-1)
            if mask.size != 1:
                return np.expand_dims(np.swapaxes(mask, 0, 1), axis=0)
            else:
                return np.ones(dep.shape, dtype=np.uint8)
