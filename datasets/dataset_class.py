import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy.ma as ma
import numpy as np
from image_utils import (EnhancedCompose, Merge, RandomCropNumpy, Split, to_tensor,
                         BilinearResize, CenterCropNumpy, RandomRotate, AddGaussianNoise,
                         RandomFlipHorizontal, RandomColor, RandomAffineZoom)
from torchvision.transforms import Lambda, Normalize, ToTensor

NYUD_MEAN = [0.48056951, 0.41091299, 0.39225179]
NYUD_STD = [0.28918225, 0.29590312, 0.3093034]

out_size = (256, 208)


class NYUDepthV2(Dataset):
    def __init__(self, imgs, test=False):
        self.imgs = imgs
        self.transform = self.get_transform(training=True, size=(256, 208))
        self.trans = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.CenterCrop((228, 304)),
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
            transforms.Resize((228, 304)),
            transforms.ToTensor(),
        ])
        self.normalize = transforms.Compose([
            transforms.Normalize(NYUD_MEAN, NYUD_STD)
        ])
        self.test = test

        if test:
            self.images = np.load('test_set/images.npy')
            self.depths = np.load('test_set/depths.npy')

    def __getitem__(self, index):
        if ~self.test:
            index = index * 10
            img_rgb = cv2.imread(self.imgs[index][0] + '/' + self.imgs[index][1])
            img_rgb = cv2.resize(img_rgb, out_size, interpolation=cv2.INTER_AREA)

            img_dep = cv2.imread(self.imgs[index][0] + '/' + self.imgs[index][2], cv2.IMREAD_GRAYSCALE)
            img_dep = cv2.resize(img_dep, out_size, interpolation=cv2.INTER_AREA)

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
            image = self.normalize(self.transform_test(self.images[index]))
            depth = self.transform_test(self.depths[index])
            return image, depth, 1

    def __len__(self):
        return int(len(self.imgs) / 10)

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
