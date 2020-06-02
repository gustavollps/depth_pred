import cv2
from cnn.cnn_mult import *
import pickle
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy.ma as ma
import numpy as np
from image_utils import (EnhancedCompose, Merge, RandomCropNumpy, Split, to_tensor,
                         BilinearResize, CenterCropNumpy, RandomRotate, AddGaussianNoise,
                         RandomFlipHorizontal, RandomColor, RandomAffineZoom)
from torchvision.transforms import Lambda, Normalize, ToTensor
import csv


def get_time(name):
    if name[0] == 'a':
        return int(name[name[2:].find('-') + 3:len(name) - 6])
    else:
        return int(name[name[2:].find('-') + 3:len(name) - 5])


NYUD_MEAN = [0.48056951, 0.41091299, 0.39225179]
NYUD_STD = [0.28918225, 0.29590312, 0.3093034]

width = 400
out_size = (width, int(width * 0.75))
BASE_PATH = "/home/gustavo/pytorch/NYU V2/nyu_data/"
PATH = '/home/gustavo/pytorch/NYU V2/nyu_data/data/nyu2_train'
CSV_TRAIN = "/home/gustavo/pytorch/NYU V2/nyu_data/data/nyu2_train.csv"
CSV_TEST = "/home/gustavo/pytorch/NYU V2/nyu_data/data/nyu2_test.csv"


class NYUDepthV2(Dataset):
    def __init__(self, test=False, shrink_factor=1):
        self.shrink_factor = shrink_factor
        self.train_list = pickle.load(open(PATH + '_list', 'rb'))  # [row for row in csv.reader(open(CSV_TRAIN))]
        self.test_list = [row for row in csv.reader(open(CSV_TEST))]
        # self.transform = self.get_transform(training=True, size=(256, 208))
        self.trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((out_size[1] - 20, out_size[0] - 20)),
            transforms.ToTensor(),
            transforms.Normalize(NYUD_MEAN, NYUD_STD)
        ])
        self.trans_depth = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            #            transforms.Normalize([0.5], [0.5])
        ])
        self.trans_mask = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
        self.transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((out_size[1] - 20, out_size[0] - 20)),
            transforms.ToTensor(),
        ])
        self.normalize = transforms.Compose([
            transforms.Normalize(NYUD_MEAN, NYUD_STD)
        ])
        self.test = test

    def __getitem__(self, index):
        if ~self.test:
            index = index * self.shrink_factor
            img_rgb = cv2.imread(self.train_list[index][0])
            img_rgb = cv2.resize(img_rgb, out_size, interpolation=cv2.INTER_AREA)

            img_dep = cv2.imread(self.train_list[index][1], cv2.IMREAD_GRAYSCALE)
            img_dep = cv2.resize(img_dep, out_size, interpolation=cv2.INTER_NEAREST)

            # depth = np.expand_dims(img_dep, axis=2)
            # temp_mask = self.get_mask(depth)
            # image, depth_mask = self.transform([img_rgb, depth, temp_mask])
            # depth = depth_mask[0].unsqueeze(0)
            # mask = depth_mask[1].unsqueeze(0)

            image = self.trans(img_rgb)
            depth = self.trans_depth(img_dep)
            mask = self.trans_mask(self.get_mask(img_dep))

            return image, depth, mask
        else:
            image = self.normalize(self.transform_test(self.test_list[index][0]))
            depth = self.transform_test(self.test_list[index][1])
            return image, depth, 1

    def __len__(self):
        if self.test:
            return int(len(self.test_list) / self.shrink_factor)
        else:
            return int(len(self.train_list) / self.shrink_factor)

    def get_mask(self, img_dep):
        img_mask_1 = ma.masked_not_equal(img_dep, 0)
        img_mask_2 = ma.masked_not_equal(img_dep, 255)
        img_mask = ~img_mask_1.mask + ~img_mask_2.mask
        img_mask = np.array(img_mask, dtype=np.uint8)
        return 255 - img_mask * 255

    def get_transform(self, training=True, size=(256, 192), normalize=True):
        if training:
            transforms = [
                Merge(),
                RandomFlipHorizontal(),
                RandomRotate(angle_range=(-5, 5), mode='constant'),
                RandomCropNumpy(size=size),
                RandomAffineZoom(scale_range=(1.0, 1.5)),
                Split([0, 3], [3, 5]),  #
                # Note: ToTensor maps from [0, 255] to [0, 1] while to_tensor does not
                [RandomColor(multiplier_range=(0.8, 1.2)), None],
            ]
        else:
            transforms = [
                [BilinearResize(0.5), None],
            ]

        transforms.extend([
            # Note: ToTensor maps from [0, 255] to [0, 1] while to_tensor does not
            [ToTensor(), Lambda(to_tensor)],
            [Normalize(mean=NYUD_MEAN, std=NYUD_STD), None] if normalize else None
        ])

        return EnhancedCompose(transforms)


def extract_number(string):
    return int(str[0:string.find('.')])


if __name__ == "__main__":
    dataset_dir = PATH
    folders = [os.path.join(dataset_dir, o) for o in os.listdir(dataset_dir)
               if os.path.isdir(os.path.join(dataset_dir, o))]
    files_list = []
    for paths in folders:
        files = os.listdir(paths)
        files_num = len(files) / 2
        if files_num - int(files_num) != 0:
            raise Exception('Odd number of files (no pairing for depth-rgb)')
        files_num = int(files_num)
        files_list = files_list + [[paths + '/' + str(i) + '.jpg', paths + '/' + str(i) + '.png']
                                   for i in range(files_num)]
    with open(PATH + '_list', 'wb') as file:
        pickle.dump(files_list, file)


def nyu_depth_old():
    dataset_dir = '/home/gustavo/pytorch/NYU V2/nyu_data/data/nyu2_train'
    folders = [os.path.join(dataset_dir, o) for o in os.listdir(dataset_dir)
               if os.path.isdir(os.path.join(dataset_dir, o))]

    subfolders = []
    for paths in folders:
        sub_subfolders = [os.path.join(paths, o) for o in os.listdir(paths)
                          if os.path.isdir(os.path.join(paths, o))]
        subfolders = subfolders + sub_subfolders

    subfolders = ['/home/gustavo/pytorch/NYU Depth V2/living_rooms_part2/living_room_0037',
                  '/home/gustavo/pytorch/NYU Depth V2/living_rooms_part2/living_room_0035']
    dataset = []

    for folder in subfolders:
        if os.path.exists(folder + '/index.txt'):
            index_file = open(folder + '/index.txt', 'r')
        elif os.path.exists(folder + '/INDEX.txt'):
            index_file = open(folder + '/INDEX.txt', 'r')
        else:
            continue

        line = index_file.readline()
        accel_files = []
        depth_files = []
        rgb_files = []

        while line:
            if line[0] == 'a':
                accel_files.append(line[:len(line) - 1])
            elif line[0] == 'd':
                depth_files.append(line[:len(line) - 1])
            elif line[0] == 'r':
                rgb_files.append(line[:len(line) - 1])
            line = index_file.readline()

        accel_index = 0
        rgb_index = 0
        old_index = 0
        debug = False
        # set dataset rgb-depth pairs
        for i in range(len(depth_files)):
            try:
                depth_time = get_time(depth_files[i])
            except:
                print("fault INDEX")
                continue

            if rgb_index == len(rgb_files) - 1:
                break

            found = True
            # set index to the rgb frame right after the current depth map
            while get_time(rgb_files[rgb_index]) < depth_time:
                rgb_index += 1
                if rgb_index >= len(rgb_files) - 1:
                    found = False
                    break

            if found:
                if rgb_index != old_index:
                    dataset.append([folder, rgb_files[rgb_index], depth_files[i]])
                    if debug:
                        print(rgb_files[rgb_index], depth_files[i])
                old_index = rgb_index

    show = True
    if show:
        for i in range(0, len(dataset), 10):
            cv2.imshow('rgb', cv2.imread(dataset[i][0] + '/' + dataset[i][1]))
            cv2.imshow('depth', cv2.imread(dataset[i][0] + '/' + dataset[i][2]))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    print(str(len(dataset)) + " images")

    with open('dataset_list.pickle', 'wb') as file:
        pickle.dump(dataset, file)
