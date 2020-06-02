from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torch.optim as optim
import losses


class multiCNN(torch.nn.Module):

    # Our batch shape for input x is (3, 32, 32)

    def __init__(self):
        super(multiCNN, self).__init__()

        # Input channels = 3, output channels = 18
        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(18, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 120, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(120, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(64, 10, kernel_size=3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(10, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = F.relu(x)
        return x


def outputSize(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2 * padding) / stride) + 1
    return output


def createLossAndOptimizer(net, learning_rate=0.001):
    # Loss function
    # loss = torch.nn.MSELoss()
    loss = losses.HuberLoss()

    # Optimizer
    self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               weight_decay=1e-4)
    return loss, optimizer


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
