import torch


def softplus(x):
    return torch.log(torch.exp(x) + 1)


def inv_softplus(x):
    return torch.log(torch.exp(x) - 1)


def diagonal_softplus(x):
    ret = x.clone()
    for i in range(x.shape[0]):
        ret[i, i] = softplus(x[i, i])
    return ret


def diagonal_inv_softplus(x):
    ret = x.clone()
    for i in range(x.shape[0]):
        ret[i, i] = inv_softplus(x[i, i])
    return ret