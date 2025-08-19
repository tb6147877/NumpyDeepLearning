import numpy as np

from framework.layers.base import Layer
from framework.utils.objectives import SoftmaxCategoricalCrossEntropy
from framework.utils.optimizers import SGD
from framework.utils.random import get_dtype
from framework.utils.random import get_rng
import framework.utils.optimizers as optimizers
import framework.utils.objectives as objectives

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

                params=[]
                grads=[]
                for layer in self.layers:
                    params+=layer.params
                    grads+=layer.grads

                self.optimizer.update(params,grads)

                train_losses.append(self.loss.forward(y_pred,y_batch))
                train_predicts.extend(y_pred)
                train_targets.extend(y_batch)

            runout = "iter %d, train-[loss %.4f, acc %.4f];"%(iter_idx, float(np.mean(train_losses)),float(self.accuracy(train_predicts,train_targets)))

            if valid_X is not None and valid_Y is not None:
                valid_losses, valid_predicts, valid_targets = [], [], []
                for b in range(valid_X.shape[0] // batch_size):
                    batch_begin = b * batch_size
                    batch_end = batch_begin + batch_size
                    x_batch = valid_X[batch_begin:batch_end]
                    y_batch = valid_Y[batch_begin:batch_end]

                    y_pred = self.predict(x_batch)

                    valid_losses.append(self.loss.forward(y_pred, y_batch))
                    valid_predicts.extend(y_pred)
                    valid_targets.extend(y_batch)

                runout = "valid-[loss %.4f, acc %.4f];" % (
                 float(np.mean(valid_losses)), float(self.accuracy(valid_predicts, valid_targets)))
            print(runout)

    def predict(self, X):
        x_next=X
        for layer in self.layers[:]:
            x_next = layer.forward(x_next)
        y_pred = x_next
        return y_pred

    def accuracy(self, outputs, targets):
        y_predicts = np.argmax(outputs,axis=1)
        y_targets = np.argmax(targets,axis=1)
        acc = y_predicts==y_targets
        return np.mean(acc)

