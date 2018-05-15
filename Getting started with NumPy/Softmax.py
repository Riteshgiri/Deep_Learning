import numpy as np


def softmax(x):

    x_exp=np.exp(x)
    x_sum=np.sum(x_exp,axis=1,keepdims=True)
    x=x_exp/x_sum
    return x

x = np.array([[1,2,3],
             [4,5,6],
             [7,8,9]])
print("softmax(x) = " + str(softmax(x)))


