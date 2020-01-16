"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import entropy, information_gain, gini_index

np.random.seed(42)

class DecisionTree():
    def __init__(self, criterion, max_depth):
        """
        Put all infromation to initialize your tree here.
        Inputs:
        > criterion : {"information_gain", "gini_index"}
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.node={}
        
    def split(self):

    def fit(self, X, y):
        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        if "float" not in str(type(y.iloc[0])):
            if self.max_depth==0:
                d={}
                for i in range(len(y)):
                    if y[i] in d:
                        d[y[i]]+=1
                    else:
                        d[y[i]]=1
            return {'output' : max(d, key=d.get)}
            if "float" not in str(type(X.iloc[0,0])):
                l=list(X.columns)
                a=[]
                mode = ''
                if self.criterion == "information_gain":
                    for i in range(len(l)):
                        a.append(information_gain(y,X[l[i]]))
                    mode = l[a.index(max(a))]
                elif self.criterion == "gini_index":
                    for i in range(l):
                        a.append(information_gain(y,X[l[i]]))
                    mode = l[a.index(max(a))]
                

                else:
                    q = set(X[mode])
                    self.node[mode] = []
                    for i in q:
                        Xd = X.where(X[mode] == i)
                        Xd.dropna(inplace = True)
                        Xd.drop([mode], axis=1, inplace=True)
                        indlist = Xd.index.tolist()
                        yd = pd.Series(y[i] for i in indlist)
                        self.node[mode].append(i)
                        self.node[mode].append(DecisionTree(self.criterion, self.max_depth-1).fit(Xd,yd))

            else:
                l=list(X.columns)
                s=[]
                for i in range(len(l)):
                    indexlist = list(X[l[i]].argsort())
                    X_sorted = X[l[i]].sort()
                    y_sorted = [y[j] for j in indexlist]
                    x = []
                    for k in range(len(X[l[i]])):
                        
                
                   

    def predict(self, X):
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        

    def plot(self):
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        
