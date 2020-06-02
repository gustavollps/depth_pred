import numpy as np
import numpy.ma as ma
import cv2
import scipy.io
import os
import matplotlib.pyplot as plt
import matplotlib
import torch

from cnn.inception_resnet_v1 import InceptionResNetV1
from cnn.inception_resnet_v2 import InceptionResNetV2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = InceptionResNetV2().to(device)
input = torch.rand((1, 3, 320, 240)).to(device)
output = net(input)
