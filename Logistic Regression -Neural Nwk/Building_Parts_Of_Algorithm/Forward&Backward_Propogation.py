
def propogate(w,b,X,Y):
    m=X.shape[1];

    # FORWARD PROPAGATION (FROM X TO COST)

    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    cost = (-1. / m) * np.sum((Y * np.log(A) + (1 - Y) * np.log(1 - A)), axis=1)  # compute cost



    # BACKWARD PROPAGATION (TO FIND GRAD)

    dw = (1. / m) * np.dot(X, ((A - Y).T))
    db = (1. / m) * np.sum(A - Y, axis=1)


    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost
