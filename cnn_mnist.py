import  os
import numpy as np
from sklearn.datasets import fetch_openml
from framework.utils.model import Model
from framework.layers.core import Dense
from framework.layers.shape import Flatten
from framework.layers.pooling import MeanPooling
from framework.layers.convolution import Convolution
from framework.utils.data import one_hot


def get_data():
    nb_data=1000
    print("loading data, please wait..")
    mnist = fetch_openml('mnist_784', data_home=os.path.join(os.path.dirname(__file__),'./data'),as_frame=False,parser='liac-arff')
    print("data loaded")

    X_train = mnist.data.reshape((-1,1,28,28))/255.0
    X_train = X_train[:nb_data]
    y_train = mnist.target
    y_train = y_train[:nb_data]

    n_classes = np.unique(y_train).size

    return n_classes,X_train,y_train

def main(max_iter):
    n_classes, X_train, y_train = get_data()

    print("building model ...")

    model=Model()
    model.add(Convolution(1, (3, 3), input_shape=(None,1,28,28),pad=1))
    model.add(MeanPooling((2, 2)))
    model.add(Convolution(2,(4,4),pad=1))
    model.add(MeanPooling((2,2)))
    model.add(Flatten())
    model.add(Dense(n_out=n_classes, activation="softmax"))
    model.compile(loss='SoftmaxCategoricalCrossEntropy', optimizer='sgd')

    print("training ...")
    model.fit(X_train,one_hot(y_train),max_iter=max_iter,validation_split=0.1)

if __name__=='__main__':
    main(10)