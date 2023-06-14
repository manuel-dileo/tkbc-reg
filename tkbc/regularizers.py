# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch
from torch import nn


class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass

class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)
        return norm / factors[0].shape[0]


class Lambda3(Regularizer):
    def __init__(self, weight: float):
        super(Lambda3, self).__init__()
        self.weight = weight

    def forward(self, factor):
        ddiff = factor[1:] - factor[:-1]
        rank = int(ddiff.shape[1] / 2)
        diff = torch.sqrt(ddiff[:, :rank]**2 + ddiff[:, rank:]**2)**3
        return self.weight * torch.sum(diff) / (factor.shape[0] - 1)

class L1(Regularizer):
    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight

    def forward(self, factor: Tuple[torch.Tensor]):
        ddiff = factor[1:] - factor[:-1]
        diff = torch.abs(ddiff)
        return self.weight * torch.sum(diff) / (factor.shape[0]-1)

class L2(Regularizer):
    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight

    def forward(self, factor: Tuple[torch.Tensor]):
        ddiff = factor[1:] - factor[:-1]
        diff = (torch.sum(torch.abs(ddiff)**2))**1/2
        return self.weight * torch.sum(diff) / (factor.shape[0]-1)

class F2(Regularizer):
    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight

    def forward(self, factor: Tuple[torch.Tensor]):
        ddiff = factor[1:] - factor[:-1]
        diff = ddiff**2
        return self.weight * torch.sum(diff) / (factor.shape[0]-1)

class N3Temp(Regularizer):
    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight

    def forward(self, factor: Tuple[torch.Tensor]):
        ddiff = factor[1:] - factor[:-1]
        diff = torch.abs(ddiff)**3
        return self.weight * torch.sum(diff) / (factor.shape[0]-1)
