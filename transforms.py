import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from draw import draw_tensor


class Rotate:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        return TF.rotate(img, self.angle)


class Flip:
    def __call__(self, img):
        if np.random.random() > 0.5:
            return TF.hflip(img)
        return img


class CenterCrop:
    def __init__(self, size):
        zoom = 1 - (np.random.random() * 0.2)

        self.resize = transforms.Resize(size)
        self.crop = transforms.CenterCrop((size[0] * zoom, size[1] * zoom))

    def __call__(self, img):
        return self.resize(self.crop(img))


class RandCrop:
    # torchvision.transforms.functional.crop(img, top, left, height, width)
    def __init__(self, size, factor):
        self.size = size
        self.factor = factor

    def __call__(self, img):
        size_factor = (1 - (np.random.random() * self.factor))
        height = size_factor * self.size[0]
        width = size_factor * self.size[1]
        # print(height/width, self.size[1]/self.size[0])
        left = int((np.random.random()) * (self.size[1] - width))
        top = int((np.random.random()) * (self.size[0] - height))
        output = TF.crop(img, left, top, int(height), int(width))
        return TF.resize(output, self.size)


def random_transform(img, size):
    return transforms.Compose([
        transforms.ToPILImage(),
        Flip(),
        RandCrop(size, 0.3),
        transforms.RandomRotation(5),
        # transforms.ColorJitter(brightness=0.2, saturation=0.2, hue=0.15),
        transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2),
        transforms.ToTensor()
    ])(img)


def sequential_transform(index, size, img):
    if index == 0:
        return transforms.ToTensor()(
            transforms.ToPILImage()(img)
        )

    elif index == 1:
        return transforms.Compose([
            transforms.ToPILImage(),
            Flip(),
            Rotate(5),
            transforms.ToTensor()
        ])(img)

    elif index == 2:
        return transforms.Compose([
            transforms.ToPILImage(),
            Flip(),
            Rotate(-5),
            transforms.ToTensor()
        ])(img)

    elif index == 3:
        return transforms.Compose([
            transforms.ToPILImage(),
            Rotate(3),
            transforms.ToTensor()
        ])(img)

    elif index == 4:
        return transforms.Compose([
            transforms.ToPILImage(),
            Rotate(-3),
            transforms.ToTensor()
        ])(img)

    elif index == 5:
        return transforms.Compose([
            transforms.ToPILImage(),
            Flip(),
            Rotate(5),
            CenterCrop(0.9, size),
            transforms.ToTensor()
        ])(img)

    elif index == 6:
        return transforms.Compose([
            transforms.ToPILImage(),
            Flip(),
            Rotate(-5),
            CenterCrop(0.9, size),
            transforms.ToTensor()
        ])(img)

    elif index == 7:
        return transforms.Compose([
            transforms.ToPILImage(),
            Rotate(5),
            CenterCrop(0.9, size),
            transforms.ToTensor()
        ])(img)

    elif index == 8:
        return transforms.Compose([
            transforms.ToPILImage(),
            Rotate(-5),
            CenterCrop(0.9, size),
            transforms.ToTensor()
        ])(img)

    elif index == 9:
        return transforms.Compose([
            transforms.ToPILImage(),
            Rotate(3),
            CenterCrop(0.9, size),
            transforms.ToTensor()
        ])(img)

    elif index == 10:
        return transforms.Compose([
            transforms.ToPILImage(),
            Rotate(-3),
            CenterCrop(0.9, size),
            transforms.ToTensor()
        ])(img)

    elif index == 11:
        return transforms.Compose([
            transforms.ToPILImage(),
            Rotate(4),
            CenterCrop(0.85, size),
            transforms.ToTensor()
        ])(img)

    elif index == 12:
        return transforms.Compose([
            transforms.ToPILImage(),
            Rotate(-4),
            CenterCrop(0.85, size),
            transforms.ToTensor()
        ])(img)

    elif index == 13:
        return transforms.Compose([
            transforms.ToPILImage(),
            Flip(),
            Rotate(4),
            CenterCrop(0.85, size),
            transforms.ToTensor()
        ])(img)

    elif index == 14:
        return transforms.Compose([
            transforms.ToPILImage(),
            Flip(),
            Rotate(-4),
            CenterCrop(0.85, size),
            transforms.ToTensor()
        ])(img)
