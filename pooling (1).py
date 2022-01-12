import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter as P


class MAC(nn.Module):
    def __init__(self):
        super(MAC, self).__init__()

    def forward(self, x):
        return F.max_pool2d(x, (x.size(-2), x.size(-1)))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class SPoC(nn.Module):
    def __init__(self):
        super(SPoC, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, (x.size(-2), x.size(-1)))

    def __repr__(self):
        return self.__class__.__name__ + '()'

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = P(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps), (x.size(-2), x.size(-1))).pow(1. / self.p)
        # return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1))) # alternative

    def __repr__(self):
        return self.__class__.__name__ + '()'
