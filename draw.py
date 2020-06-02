import matplotlib.pyplot as plt
import numpy as np
import cv2
import numpy
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import PIL.Image

def plot2cv2(data):
    image = None
    for i in data:
        fig = plt.figure()
        plot = fig.add_subplot(111)

        plot.plot(i)

        fig.canvas.draw()
        output = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        output = output.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        output = cv2.resize(output, (int(output.shape[1] / 2), int(output.shape[0] / 2)))
        if image is not None:
            image = np.concatenate([image, output], axis=1)
        else:
            image = output

    plt.close('all')
    cv2.imwrite('./checkpoints/latest_training.jpg', image)
    return image


def draw_tensor(tensor, mask=None):
    if isinstance(tensor, numpy.ndarray):
        tensor = transforms.ToTensor()(tensor)
    elif isinstance(tensor, PIL.Image.Image):
        tensor = transforms.ToTensor()(tensor)
        tensor = tensor.permute(1, 0, 2)
    elif ~isinstance(tensor, torch.Tensor):
        return

    if len(tensor.shape) < 4:
        tensor = tensor.unsqueeze(0)

    if tensor.shape[2] == 1:
        rgb_tensor = False
    else:
        rgb_tensor = True
    print(tensor.shape, rgb_tensor)

    if mask is not None:
        msk = np.swapaxes(
            np.swapaxes(mask.cpu().numpy()[0], 0, 1)
            , 1, 2)
        tmp_mask = msk
        msk = cv2.cvtColor((255 - msk * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
        tensor = tensor * msk

    if rgb_tensor is True:
        out = tensor.cpu().numpy()[0]
        out = np.swapaxes((out * 255).astype('uint8'), 1, 2)
    else:
        out = tensor.cpu().numpy()[0]
        out = (out - out.min()) / (out.max() - out.min()) * 255
        out = (out - out.min()) / (out.max() - out.min()) * 255
        print("depth shape", out.shape)
        out = cv2.applyColorMap((np.swapaxes(out, 2, 1)).astype('uint8'), cv2.COLORMAP_JET)

    if mask is not None:
        out = out * tmp_mask

    print(out.shape)
    cv2.imshow('debug', out)
    cv2.waitKey(2000)


def draw_debug(inputs, labels, mask, outputs, name='training'):
    rgb = np.swapaxes((inputs * mask).cpu().numpy()[0], 0, 2)
    rgb = np.swapaxes((rgb * 255).astype('uint8'), 0, 1)

    msk = np.swapaxes(
        np.swapaxes(mask.cpu().numpy()[0], 0, 1)
        , 1, 2)
    tmp_mask = msk
    msk = cv2.cvtColor((255 - msk * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)

    lbl = (np.swapaxes((labels * mask).cpu().numpy()[0], 0, 2))
    # lbl = (np.swapaxes((labels).cpu().numpy()[0], 0, 2))
    out = np.swapaxes(((outputs * mask).cpu().detach().numpy()[0]), 1, 2)

    out = (out - lbl.min()) / (lbl.max() - lbl.min()).clip(min=0) * 255
    lbl = (lbl - lbl.min()) / (lbl.max() - lbl.min()) * 255

    lbl = cv2.applyColorMap((np.swapaxes(lbl, 0, 1)).astype('uint8'), cv2.COLORMAP_JET)
    lbl = lbl * tmp_mask
    out = cv2.applyColorMap(np.swapaxes(out, 0, 2).astype('uint8'), cv2.COLORMAP_JET)
    out = out * tmp_mask
    out = cv2.resize(np.concatenate([rgb, lbl, out, msk], axis=1), (170 * 8, 223 * 2))
    cv2.imshow(name, out)
    if name == 'training':
        pos = 300
    else:
        pos = 800
    cv2.moveWindow(name, 300, pos)
    cv2.waitKey(10)
