from framework.layers.base import Layer


class Linear(Layer):
    def __init__(self, n_out, n_in=None, init='glorot_uniform'):
        self.n_out = n_out
        self.n_in = n_in
        self.out_shape = (None, n_out)
