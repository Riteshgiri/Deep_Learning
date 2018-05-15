import numpy as np
import math

def sigmoid_derivative(x):

    s=1/(1+np.exp(-x))
    #see the difference between np.dot and simple multiply
    # np.dot using matrix multiplication.
    ds1=np.dot(s,(1-s))
    ds2=s*(1-s)
    return ds1,ds2

x=np.array([1,2,3])
print(sigmoid_derivative(x))

