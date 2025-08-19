import numpy as np

from framework.layers.base import Layer
from framework.utils.initializations import _zero
from framework.utils.initializations import get as get_init
from framework.utils import activations

class Convolution(Layer):

    def __init__(self,nb_filter,filter_size,input_shape=None,stride=1,pad=0,init='glorot_uniform',activation='relu'):
        self.nb_filter = nb_filter
        self.filter_size = filter_size
        self.input_shape = input_shape
        self.stride = stride
        self.pad = pad

        self.W = None
        self.b = None
        self.dW = None
        self.db = None
        self.out_shape = None
        self.last_output = None
        self.last_input = None

        self.init = get_init(init)
        self.activation = activations.get(activation)

    def connect_to(self, prev_layer=None):
        if prev_layer is None:
            assert self.input_shape is not None
            input_shape = self.input_shape
        else:
            input_shape = prev_layer.out_shape

        # input_shape: (batch size, num input feature maps, image height, image width)
        assert len(input_shape) == 4

        nb_batch, pre_nb_filter,pre_height,pre_width = input_shape

        filter_height,filter_width = self.filter_size

        height = (pre_height-filter_height+2*self.pad)//self.stride+1
        width = (pre_width-filter_width+2*self.pad)//self.stride+1

        self.out_shape = (nb_batch,self.nb_filter,height,width)

        self.W = self.init((self.nb_filter,pre_nb_filter,filter_height,filter_width))
        self.b = _zero((self.nb_filter,))