import math
import torch


def cauchy_log_prob(x, scale):
    if scale > 0.:
        return math.log(scale / math.pi / 2) - torch.log(x ** 2 + scale ** 2)
    else:
        return - torch.log(x ** 2)
