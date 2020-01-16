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
        
    def split(self, X, y, attr):
        sortind = X[attr].argsort()
        X_sort = X[attr].sort()
        y_sort = [y[i] for i in sortind]
        s=[]
        m=[]
        n=[]
        for i in range(len(X_sort)-1):
            a = (X_sort[i]+X_sort[i+1])/2
            for j in range(len(X)):
                if X_sort[j]<=a:
                    m.append(y_sort[j])
                else:
                    n.append(y_sort[j])
            s.append(entropy(y) - (len(m)/len(y))*entropy(m) - (len(n)/len(y))*entropy(n))
        return s


    def getnd(self, X, y):
        l = list(X.columns)
        a = []
        p=0
        for i in range(len(l)):
            d={}
            q = set(X[l[i]])
            for j in X[l[i]]:
                if j in d:
                    d[j].append(y[j])
                else:
                    d[j] = [y[j]]
            for k in d:
                p+=(len(d[k])/len(y))*np.std(np.array(d[k]))**2
            totvar = np.std(y)**2 - p
            a.append(totvar)
        ind = a.index(max(a))
        return l[ind]

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
                

                if max_depth!=0:
                    q = set(X[mode])
                    self.node[mode] = []
                    for i in q:
                        Xd = X.loc[X[mode]==i].copy()
                        Xd.drop([mode],axis=1,inplace=True)
                        indlist = Xd.index.tolist()
                        yd = pd.Series(y[i] for i in indlist)
                        self.node[mode].append(i)
                        self.node[mode].append(DecisionTree(self.criterion, self.max_depth-1).fit(Xd,yd))

            else:
                h = list(X.columns)
                s=[]
                for i in h:
                    s+=DecisionTree(self.criterion, self.max_depth).split(X,y,i)
                maxind = [(s.index(max(s))//len(y))-1, s.index(max(s)%len(y) - 1)]
                div_node = h[maxind[0]]
                div_ind = maxind[1]
                X_sort = X[div_node].sort()
                div_val = (X_sort[div_ind]+X_sort[div_ind+1])/2
                Xd1 = X.loc[X[mode] <= div_val]
                Xd2 = X.loc[X[mode] > div_val]
                Xd1.drop([mode],axis=1,inplace=True)
                Xd2.drop([mode],axis=1,inplace=True)
                indlist1 = Xd1.index.tolist()
                yd1 = pd.Series(y[i] for i in indlist)
                indlist2 = Xd2.index.tolist()
                yd2 = pd.Series(y[i] for i in indlist) 
                self.node[div_node]=[]
                self.node[div_node].append("<= "+ str(div_val))
                self.node[div_node].append(DecisionTree(self.criterion, self.max_depth-1).fit(Xd1,yd1))
                self.node[div_node].append("> "+ str(div_val))
                self.node[div_node].append(DecisionTree(self.criterion, self.max_depth-1).fit(Xd2,yd2)) 

        else:
            if self.max_depth==0:
                return {"output":sum(y)/y.size}


            if "float" not in str(type(y.iloc[0])):
                if max_depth!=0:
                    mode = DecisionTree(self.criterion, self.max_depth).getnd(X,y)
                    q = set(X[mode])
                    self.node[mode] = []
                    for i in q:
                        Xd = X.loc[X[mode]==i].copy()
                        Xd.drop([mode],axis=1,inplace=True)
                        indlist = Xd.index.tolist()
                        yd = pd.Series(y[i] for i in indlist)
                        self.node[mode].append(i)
                        self.node[mode].append(DecisionTree(self.criterion, self.max_depth-1).fit(Xd,yd))

            else:
                h = list(X.columns)
                s=[]
                for i in h:
                    s+=DecisionTree(self.criterion, self.max_depth).split(X,y,i)
                maxind = [(s.index(max(s))//len(y))-1, s.index(max(s)%len(y) - 1)]
                div_node = h[maxind[0]]
                div_ind = maxind[1]
                X_sort = X[div_node].sort()
                div_val = (X_sort[div_ind]+X_sort[div_ind+1])/2
                Xd1 = X.loc[X[mode] <= div_val]
                Xd2 = X.loc[X[mode] > div_val]
                Xd1.drop([mode],axis=1,inplace=True)
                Xd2.drop([mode],axis=1,inplace=True)
                indlist1 = Xd1.index.tolist()
                yd1 = pd.Series(y[i] for i in indlist)
                indlist2 = Xd2.index.tolist()
                yd2 = pd.Series(y[i] for i in indlist) 
                self.node[div_node]=[]
                self.node[div_node].append("<= "+ str(div_val))
                self.node[div_node].append(DecisionTree(self.criterion, self.max_depth-1).fit(Xd1,yd1))
                self.node[div_node].append("> "+ str(div_val))
                self.node[div_node].append(DecisionTree(self.criterion, self.max_depth-1).fit(Xd2,yd2))         

        return self.node
                


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
        
