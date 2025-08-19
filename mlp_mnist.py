import  os
import numpy as np
from sklearn.datasets import fetch_openml
from framework.utils.model import Model
from framework.layers.core import Dense
from framework.utils.data import one_hot


def get_data():
    print("loading data, please wait..")
    mnist = fetch_openml('mnist_784', data_home=os.path.join(os.path.dirname(__file__),'./data'),as_frame=False,parser='liac-arff')
    print("data loaded")

    X_train = mnist.data/255.0
    y_train = mnist.target

    n_classes = np.unique(y_train).size

    return n_classes,X_train,y_train

def main(max_iter):
    n_classes, X_train, y_train = get_data()

    print("building model ...")

    model=Model()
    model.add(Dense(n_out=200,n_in=784,activation="relu"))
    model.add(Dense(n_out=n_classes, activation="softmax"))
    model.compile(loss='SoftmaxCategoricalCrossEntropy', optimizer='sgd')

    print("training ...")
    model.fit(X_train,one_hot(y_train),max_iter=max_iter,validation_split=0.1)

if __name__=='__main__':
    main(50)

