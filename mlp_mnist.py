import  os
import numpy as np
from sklearn.datasets import fetch_mldata


def get_data():
    print("loading data, please wait..")
    mnist = fetch_mldata('MNIST original', data_home=os.path.join(os.path.dirname(__file__),'./data'))
