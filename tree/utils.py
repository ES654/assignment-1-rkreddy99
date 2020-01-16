import math
import pandas as pd
def entropy(Y):
    """
    Function to calculate the entropy 
    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the entropy as a float
    """
    d={}
    for i in range(len(Y)):
      if Y.iloc[i] in d:
        d[Y.iloc[i]]+=1
      elif Y.iloc[i] not in d:
        d[Y.iloc[i]] = 1
    sum=0
    for i in d:
      sum+=d[i]
    for i in d:
      d[i]=d[i]/sum
    entro=0
    for i in d:
      entro += d[i]*math.log(d[i],2)
    entro = -entro
    return entro
    

def gini_index(Y):
    """
    Function to calculate the gini index
    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the gini index as a float
    """
    d={}
    for i in range(len(Y)):
      if Y.iloc[i] in d:
        d[Y.iloc[i]]+=1
      elif Y.iloc[i] not in d:
        d[Y.iloc[i]] = 1
    sum=0
    for i in d:
      sum+=d[i]

    p=0
    for i in d:
      d[i]=d[i]/sum
      p+=d[i]**2

    ind = 1-p
    return ind
  
      
def information_gain(Y, attr):
    """
    Function to calculate the information gain
    
    Inputs:
    > Y: pd.Series of Labels
    > attr: attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    assert(len(Y)==len(attr))

    a = set(attr)
    d={}
    for i in range(len(attr)):
        if attr[i] in d:
            d[attr[i]].append(Y[i])
        elif attr[i] not in d:
            d[attr[i]] = [Y[i]]
    
    t=0
    for i in a:
        t+=(len(d[i])/len(Y))*entropy(pd.Series(d[i]))
    gain = entropy(Y) - t
    return gain
    