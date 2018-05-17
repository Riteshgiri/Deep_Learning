def sigmoid(z):
    """
    z -- A scalar or numpy array of any size.
 """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1 / (1 + np.exp(-z))
    ### END CODE HERE ###

    return s