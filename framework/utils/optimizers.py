import copy

import numpy as np


from framework.utils.initializations import _zero

class Optimizer():
    def __init__(self, lr = 0.001, clip=-1, decay=0.0, lr_min=0.0, lr_max=np.inf):
        self.lr = lr
        self.clip = clip
        self.decay = decay
        self.lr_min = lr_min
        self.lr_max = lr_max

        self.iterations=0


    def update(self, params, grads):

        self.iterations+=1

        self.lr *= (1.0/(1.0+self.decay*self.iterations))

        self.lr = np.clip(self.lr, self.lr_min, self.lr_max)

    def __str__(self):
        return self.__class__.__name__


class SGD(Optimizer):
    def __init__(self, *args, **kwargs):
        super(SGD, self).__init__(*args, **kwargs)

    def update(self, params, grads):
        for p,g in zip(params, grads):
            p-=self.lr*opt_clip(g,self.clip) # clip默认值是-1，所以grads clip默认不开

        super(SGD, self).update(params, grads)


def opt_clip(grad, boundary):
    if boundary>0:
        return np.clip(grad, -boundary, boundary)
    else:
        return grad


def get(optimizer):
    if optimizer.__class__.__name__ == 'str':
        if optimizer in ['sgd', 'SGD']:
            return SGD()

        raise ValueError('Unknown optimizer name: {}.'.format(optimizer))

    elif isinstance(optimizer, Optimizer):
        return copy.deepcopy(optimizer)

    else:
        raise ValueError('Unknown type: {}.'.format(optimizer.__class__.__name__))