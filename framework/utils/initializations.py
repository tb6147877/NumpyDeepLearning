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

class Uniform(Initializer):
    def __init__(self, scale=0.05):
        self.scale = scale

    def call(self,size):
        return _cast_dtype(get_rng().uniform(-self.scale,self.scale,size=size))

class Normal(Initializer):
    def __init__(self,std=0.01,mean=0.0):
        self.std = std
        self.mean = mean

    def call(self,size):
        return _cast_dtype(get_rng().normal(loc=self.mean,scale=self.std,size=size))

class GlorotUniform(Initializer):
    def call(self,size):
        fan_in,fan_out=decompose_size(size)
        return Uniform(np.sqrt(6/(fan_in+fan_out)))(size)

def _cast_dtype(res):
    return np.array(res, dtype=get_dtype())

_zero=Zero()
_one=One()

#这个函数其实就在计算各种初始化算法的缩放
def decompose_size(size):
    if len(size)==2: #全连接层
        fan_in=size[0]
        fan_out=size[1]
    elif len(size)==4 or len(size)==5: #卷积层
        respective_field_size=np.prod(size[2:])
        fan_in = size[1]*respective_field_size
        fan_out=size[0]*respective_field_size
    else:
        fan_in=fan_out=int(np.sqrt(np.prod(size)))

    return fan_in,fan_out


def get(initialization):
    if initialization.__class__.__name__=="str":
        if initialization in ['zero','Zero']:
            return Zero()
        if initialization in ['one','One']:
            return One()
        if initialization in ['uniform','Uniform']:
            return Uniform()
        if initialization in ['normal','Normal']:
            return Normal()
        if initialization in ['glorot_uniform','GlorotUniform']:
            return GlorotUniform()
        raise ValueError('Unknown initialization name: {}.'.format(initialization))
    elif isinstance(initialization, Initializer):
        return copy.deepcopy(initialization)
    else:
        raise ValueError('Unknown type: {}.'.format(initialization.__class__.__name__))