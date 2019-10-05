import numpy as np

import torch
from torch import nn


def squash_dims(tensor, dim_begin, dim_end=None):
    shape = tensor.shape
    if dim_end is None:
        dim_end = len(shape)
    if dim_begin < 0:
        dim_begin += len(shape)
        dim_end += len(shape)
    to_squash = shape[dim_begin:dim_end]
    tensor = tensor.view(*shape[:dim_begin], np.prod(to_squash), *shape[dim_end:])
    return tensor


class LSTM(nn.LSTM):
    def forward(self, *args, **kwargs):
        output, (h_n, c_n) = super().forward(*args, **kwargs)
        if self.batch_first:
            h_n = h_n.transpose(0, 1)
            c_n = c_n.transpose(0, 1)
        return output, (h_n, c_n)


class Distributed(nn.Module):
    def __init__(self, module, depth=1):
        super().__init__()
        self.module = module
        self.depth = depth

    def forward(self, x):
        shape = x.shape
        n = self.depth + 1
        shape_head, shape_tail = shape[:n], shape[n:]
        x = x.view(np.prod(shape_head), *shape_tail)
        x = self.module(x)
        x = self.view_tuple(x, *shape_head)
        return x

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.module.__repr__()})'

    @classmethod
    def view_tuple(cls, x, *shape_head):
        if isinstance(x, tuple):
            return tuple(cls.view_tuple(t, *shape_head) for t in x)
        x = x.view(*shape_head, *x.shape[1:])
        return x


def cat_with_broadcast(tensors, dim=0):
    assert len(tensors)
    assert all(len(t.shape) == len(tensors[0].shape) for t in tensors[1:])
    shapes = np.array([t.shape for t in tensors])
    target_shape = shapes.max(0)
    expanded_tensors = []
    for tensor in tensors:
        target_shape[dim] = tensor.shape[dim]
        x = tensor.expand(*target_shape)
        expanded_tensors.append(x)
    x = torch.cat(expanded_tensors, dim=dim)
    return x
