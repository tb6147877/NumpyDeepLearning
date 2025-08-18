import numpy as np

from framework.layers.base import Layer
from framework.utils.objectives import SoftmaxCategoricalCrossEntropy
from optimizers import SGD
from random import get_dtype
from random import get_rng
import optimizers
import objectives

import sys

class Model():
    def __init__(self, layers=None):
        self.layers = [] if layers is None else layers
        self.loss = None
        self.optimizer = None

    def add(self, layer):
        assert isinstance(layer, Layer), "Must be 'Layer' instance"
        self.layers.append(layer)

    def compile(self, loss=SoftmaxCategoricalCrossEntropy(), optimizer=SGD()):
        self.layers[0].first_layer = True

        next_layer=None
        for layer in self.layers:
            layer.connect_to(next_layer)
            next_layer = layer

        self.loss = objectives.get(loss)

        self.optimizer = optimizers.get(optimizer)

    def fit(self, X, Y, max_iter=100, batch_size=64, shuffle=True, validation_split=0.0, validation_data=None, file=sys.stdout):
        train_X = X.astype(get_dtype()) if np.issubdtype(np.float64, X.dtype) else X
        train_Y = Y.astype(get_dtype()) if np.issubdtype(np.float64, Y.dtype) else Y

        if 1.>validation_split>0.:
            split = int(train_Y.shape[0]*validation_split)
            valid_X, valid_Y = train_X[-split:], train_Y[-split:]
            train_X, train_Y = train_X[:-split], train_Y[:-split]

        elif validation_data is not None:
            valid_X, valid_Y = validation_data

        else:
            valid_X, valid_Y = None,None

        iter_idx = 0
        while iter_idx < max_iter:
            iter_idx += 1

            if shuffle:
                seed=get_rng().randint(100, 10000)
                np.random.seed(seed)
                np.random.shuffle(train_X)
                np.random.seed(seed)
                np.random.shuffle(train_Y)

            train_losses, train_predicts, train_targets= [], [], []
            for b in range(train_Y.shape[0]//batch_size):
                batch_begin = b*batch_size
                batch_end = batch_begin + batch_size
                x_batch = train_X[batch_begin:batch_end]
                y_batch = train_Y[batch_begin:batch_end]

                y_pred = self.predict(x_batch)

                next_grad =self.loss.backward(y_pred, y_batch)
                for layer in self.layers[::-1]:
                    next_grad = layer.backward(next_grad)



    def predict(self, X):
        x_next=X
        for layer in self.layers[:]:
            x_next = layer.forward(x_next)
        y_pred = x_next
        return y_pred

