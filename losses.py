import numpy as np
import torch
import torch.nn as nn
from math import log


def _mask_input(input, mask=None):
    if mask is not None:
        input = input * mask
        count = torch.sum(mask).item()
    else:
        count = np.prod(input.size(), dtype=np.float32).item()
    return input, count


class RMSLoss(nn.Module):
    def forward(self, input, target, mask=None):
        loss = torch.pow(input - target, 2)
        loss, count = _mask_input(loss, mask)
        return torch.sqrt(torch.sum(loss) / count)


class HuberLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.SmoothL1Loss(size_average=False)

    def forward(self, input, target, mask):
        loss = self.loss(input * mask, target * mask)
        count = torch.sum(mask).item()
        return loss / count


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, input, target, mask):
        loss = torch.sqrt(self.mse(input * mask, target * mask))
        count = torch.sum(mask).item()
        return loss / count


class RMSLoss(nn.Module):
    def forward(self, input, target, mask=None):
        loss = torch.pow(input - target, 2)
        loss, count = _mask_input(loss, mask)
        return torch.sqrt(torch.sum(loss) / count)


class RelLoss(nn.Module):
    def forward(self, input, target, mask=None):
        loss = torch.abs(input - target) / target
        loss, count = _mask_input(loss, mask)
        return torch.sum(loss) / count


class Log10Loss(nn.Module):
    def forward(self, input, target, mask=None):
        loss = torch.abs((torch.log(target) - torch.log(input)) / log(10))
        loss, count = _mask_input(loss, mask)
        return torch.sum(loss) / count


class BerHuLoss(nn.Module):
    def forward(self, input, target, mask=None):
        x = input - target
        abs_x = torch.abs(x)
        c = torch.max(abs_x).item() / 5
        leq = (abs_x <= c).float()
        l2_losses = (x ** 2 + c ** 2) / (2 * c)
        losses = leq * abs_x + (1 - leq) * l2_losses
        losses, count = _mask_input(losses, mask)
        return torch.sum(losses) / count


class SilogLoss(nn.Module):
    def __init__(self):
        super(SilogLoss, self).__init__()

    def forward(self, ip, target, ratio=10, ratio2=0.85, lim=(1, 81), mask=None):
        ip = ip.reshape(-1)
        target = target.reshape(-1)

        mask = (target > lim[0]) & (target < lim[1])
        masked_ip = torch.masked_select(ip.float(), mask)
        masked_op = torch.masked_select(target, mask)

        log_diff = torch.log(masked_ip * ratio) - torch.log(masked_op * ratio)
        log_diff_masked = log_diff

        silog1 = torch.mean(log_diff_masked ** 2)
        silog2 = ratio2 * (torch.mean(log_diff_masked) ** 2)
        silog_loss = torch.sqrt(silog1 - silog2) * ratio
        return silog_loss
