from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import math
import torch
from torch import nn
import numpy as np

from models import TKBCModel

@torch.jit.script
def normalize_phases(p_emb):
    # normalize phases so that they lie in [-pi,pi]
    # first shift phases by pi
    out = p_emb + math.pi
    # compute the modulo (result then in [0,2*pi))
    out = torch.remainder(out, 2.0 * math.pi)
    # shift back
    out = out - math.pi
    return out

@torch.jit.script
def hadamard_complex(x_re, x_im, y_re, y_im):
    """Hadamard product for complex vectors"""
    result_re = x_re * y_re - x_im * y_im
    result_im = x_re * y_im + x_im * y_re
    return result_re, result_im

class ChronoR(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2
    ):
        super(ChronoR, self).__init__()
        self.sizes = sizes
        if rank % 2 != 0:
            raise Exception('Rank must be even')

        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], 2 * rank, sparse=True),
            nn.Embedding(sizes[1], rank//2, sparse=True),
            nn.Embedding(sizes[3], rank//2, sparse=True)
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size

        self.no_time_emb = no_time_emb

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])


        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        rel = normalize_phases(rel)
        rel = torch.cos(rel), torch.sin(rel)
        time = normalize_phases(time)
        time = torch.cos(time), torch.sin(time)

        rt = torch.cat((rel[0],time[0]), 1), torch.cat((rel[1],time[1]), 1)

        hrt = hadamard_complex(lhs[0], lhs[1], rt[0], rt[1])

        return torch.sum(sum(hadamard_complex(hrt[0],hrt[1],rhs[0],rhs[1])))
    
    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        rel = normalize_phases(rel)
        rel = torch.cos(rel), torch.sin(rel)
        time = normalize_phases(time)
        time = torch.cos(time), torch.sin(time)

        rt = torch.cat((rel[0], time[0]), 1), torch.cat((rel[1], time[1]), 1)

        hrt = hadamard_complex(lhs[0], lhs[1], rt[0], rt[1])

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        return (
                       hrt[0] @ right[0].t() +
                       hrt[1] @ right[1].t()
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(rt[0] ** 2 + rt[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               ), sself.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        rel = normalize_phases(rel)
        rel = torch.cos(rel), torch.sin(rel)
        time = normalize_phases(time)
        time = torch.cos(time), torch.sin(time)

        rt = torch.cat((rel[0], time[0]), 1), torch.cat((rel[1], time[1]), 1)

        ht = hadamard_complex(lhs[0], lhs[1], rhs[0], rhs[1])

        return (
                ht @ rt[0].t() +
                ht @ rt[1].t()
        )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]

        rel = normalize_phases(rel)
        rel = torch.cos(rel), torch.sin(rel)
        time = normalize_phases(time)
        time = torch.cos(time), torch.sin(time)

        rt = torch.cat((rel[0], time[0]), 1), torch.cat((rel[1], time[1]), 1)

        hrt = hadamard_complex(lhs[0], lhs[1], rt[0], rt[1])

        return torch.cat([hrt[0], hrt[1]],1)