import numpy as np

def L1(y,yhat):
    """
        yhat -- vector of size m (predicted labels)
        y -- vector of size m (true labels)
        """
    loss = np.sum(np.dot(y - yhat, y - yhat))

    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(y,yhat)))

