import math
import numpy as np


def basic_sigmoid(x):
    """

    Compute sigmoid of x.

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1 / (1 + math.exp(-x))
    ### END CODE HERE ###

    return s


print(basic_sigmoid(3))

# Actually, we rarely use the "math" library in deep learning because the inputs of the functions are real numbers.
#  In deep learning we mostly use matrices and vectors. This is why numpy is more useful.

def basic_sigmoid_np(x):
    s=1/(1+np.exp(-x))
    return s


x=np.array([1,2,3])

print(basic_sigmoid_np(x))