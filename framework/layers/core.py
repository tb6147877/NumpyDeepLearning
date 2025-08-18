import numpy as np

from framework.layers.base import Layer
from framework.utils.initializations import _zero
from framework.utils.initializations import get as get_init
from framework.utils import activations

class Linear(Layer):
    def __init__(self, n_out, n_in=None, init='glorot_uniform'):
        self.n_out = n_out
        self.n_in = n_in
        self.out_shape = (None, n_out)
        self.init = get_init(init)

        self.W = None
        self.b = None
        self.dW=None
        self.db=None
        self.last_input=None

    def connect_to(self, prev_layer=None):
        if prev_layer is None:
            assert self.n_in is not None
            n_in = self.n_in
        else:
            assert len(prev_layer.out_shape) == 2
            n_in = prev_layer.out_shape[-1]

        self.W = self.init((n_in, self.n_out))
        self.b = _zero((self.n_out,))

    def forward(self, input, *args, **kwargs):
        self.last_input = input
        return np.dot(input, self.W) + self.b

    def backward(self, pre_grad, *args, **kwargs):
        self.dW = np.dot(self.last_input.T, pre_grad) # pre_grad就是dA[L]，self.last_input就是A[L-1]
        self.db = np.mean(pre_grad, axis=0)

        if not self.first_layer:
            return np.dot(pre_grad, self.W.T)

    @property
    def params(self):
        return self.W, self.b

    @property
    def grads(self):
        return self.dW, self.db


class Dense(Layer):
    def __init__(self, n_out, n_in=None, init='glorot_uniform',activation='tanh'):
        self.n_out = n_out
        self.n_in = n_in
        self.out_shape = (None, n_out)
        self.init = get_init(init)
        self.act_layer = activations.get(activation)

        self.W = None
        self.b = None
        self.dW = None
        self.db = None
        self.last_input = None

    def connect_to(self, prev_layer=None):
        if prev_layer is None:
            assert self.n_in is not None
            n_in = self.n_in
        else:
            assert len(prev_layer.out_shape) == 2
            n_in = prev_layer.out_shape[-1]
        self.W = self.init((n_in, self.n_out))
        self.b = _zero((self.n_out,))

    def forward(self, input, *args, **kwargs):
        self.last_input = input
        linear_out = np.dot(input, self.W) + self.b
        act_out = self.act_layer.forward(linear_out)
        return act_out

    def backward(self, pre_grad, *args, **kwargs):
        act_grad = pre_grad * self.act_layer.derivative()
        self.dW = np.dot(self.last_input.T, act_grad)
        self.db = np.mean(act_grad, axis=0)
        if not self.first_layer:
            return np.dot(act_grad, self.W.T)

    @property
    def params(self):
        return self.W, self.b

    @property
    def grads(self):
        return self.dW, self.db