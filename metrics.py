
def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    # TODO: Write here
    a=0
    for i in range(len(y)):
        if y[i]==y_hat[i]:
            a+=1
    acc = a/len(y)
    return acc
    pass

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    q=0
    w=0
    for i in range(len(y)):
        if y_hat[i]==y[i]  and y_hat[i]==cls:
            q+=1
        if y_hat[i]==cls:
            w+=1
    pre = q/w
    return pre
    pass

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    e=0
    r=0
    for i in range(len(y)):
        if y_hat[i]==y[i] and y[i]==cls:
            e+=1
        if y[i]==cls:
            r+=1
    rec = e/r
    return rec
    pass

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    rse = (sum((y_hat-y)**2)/len(y))**0.5
    return rse
    pass

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    me = sum(abs(y_hat-y))/len(y)
    return me
    pass
