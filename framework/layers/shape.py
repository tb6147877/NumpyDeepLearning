import numpy as np

from framework.layers.base import Layer

class Flatten(Layer):
    def __init__(self, outdim=2):
        self.outdim = outdim
        if outdim<1:
            raise ValueError('outdim must be > 0 , was %i', outdim)

        self.last_input_shape = None
        self.out_shape = None

    def connect_to(self,prev_layer):
        assert len(prev_layer.out_shape) > 2
        to_flatten = np.prod(prev_layer.out_shape[self.outdim-1:])
        flattened_shape = prev_layer.out_shape[:self.outdim-1]+(to_flatten,)
        self.out_shape = flattened_shape

    def forward(self, input,*args, **kwargs):
        self.last_input_shape = input.shape

        flattened_shape = input.shape[:self.outdim-1]+(-1,)
        return np.reshape(input,flattened_shape)

    def backward(self, pre_grad, *args, **kwargs):
        return np.reshape(pre_grad, self.last_input_shape)
