from framework.utils.random import get_rng
from framework.utils.random import get_dtype

import numpy as np
import copy

class Initializer():

    def __call__(self, size):
        return self.call(size)

    def call(self, size):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

class One(Initializer):
    def call(self,size):
        return _cast_dtype(np.ones(size))

class Zero(Initializer):
    def call(self,size):
        return _cast_dtype(np.zeros(size))

def _cast_dtype(res):
    return np.array(res, dtype=get_dtype())

#这个函数其实就在计算各种初始化算法的缩放
def decompose_size(size):
