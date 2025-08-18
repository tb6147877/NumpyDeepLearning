import copy

import numpy as np


class Objective():
    def forward(self,outputs,targets):
        raise NotImplementedError

    def backward(self,outputs,targets):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class SoftmaxCategoricalCrossEntropy(Objective):

    def __init__(self, epsilon=1e-11):
        self.epsilon = epsilon

    def forward(self,outputs,targets):
        outputs = np.clip(outputs,self.epsilon,1-self.epsilon)
        return np.mean(-np.sum(targets * np.log(outputs),axis=1))

    def backward(self,outputs,targets):
        outputs = np.clip(outputs,self.epsilon,1-self.epsilon)

        return outputs-targets

SCCE = SoftmaxCategoricalCrossEntropy

def get(objective):
    if objective.__class__.__name__ == 'str':
        if objective in ['softmax_categorical_cross_entropy','SoftmaxCategoricalCrossEntropy']:
            return SoftmaxCategoricalCrossEntropy()
        raise ValueError('Unknown objective name: {}.'.format(objective))

    elif isinstance(objective, Objective):
        return copy.deepcopy(objective)

    else:
        raise ValueError('Unknown type: {}.'.format(objective.__class__.__name__))