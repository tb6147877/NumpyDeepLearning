from framework.layers.base import Layer
from framework.utils.initializations import _zero
from framework.utils.initializations import get as get_init

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


