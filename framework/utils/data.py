import numpy as np

def one_hot(labels, nb_classes=None):
    classes = np.unique(labels)
    if nb_classes is None:
        nb_classes =classes.size
    one_hot_labels = np.zeros((labels.shape[0], nb_classes))
    for i,c in enumerate(classes):
        one_hot_labels[labels==c,i] = 1
    return one_hot_labels