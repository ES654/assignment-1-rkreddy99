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
        X_sort = X.sort_values(attr)
        X_sort = X_sort[attr]
        y_sort = [y.iloc[i] for i in sortind]
        s=[]
        m=[]
        n=[]
        for i in range(len(X_sort)-1):
            if X_sort.iloc[i]!=X_sort.iloc[i+1]:
                a = (X_sort.iloc[i]+X_sort.iloc[i+1])/2
                for j in range(len(X)):
                    if X_sort.iloc[j]<=a:
                        m.append(y_sort[j])
                    else:
                        n.append(y_sort[j])
                s.append(entropy(y) - (len(m)/len(y))*entropy(pd.Series(m)) - (len(n)/len(y))*entropy(pd.Series(n)))
            else:
                s.append(-9999999)
        return s


    def getnd(self, X, y):
        l = list(X.columns)
        a = []
        p=0
        for i in range(len(l)):
            d={}
            for j in X.iloc[X.columns.get_loc(l[i])]:
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
        if (X.iloc[0,0]).dtype!=float:
            if self.max_depth==0 or len(np.unique(y))==1 or len(y) == 1:
                d={}
                for i in range(len(y)):
                    if y[i] in d:
                        d[y[i]]+=1
                    else:
                        d[y[i]]=1
                return {'output' : max(d, key=d.get)}
            if y.dtype!=float:
                l=list(X.columns)
                a=[]
                mode = ''
                if self.criterion == "information_gain":
                    for i in range(len(l)):
                        a.append(information_gain(y,X[l[i]]))
                    mode = l[a.index(max(a))]

                if self.criterion == "gini_index":
                    for i in range(len(l)):
                        a.append(gini_index(y,X[l[i]]))
                    mode = l[a.index(max(a))]
                

                if self.max_depth!=0 :
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
                    s.append(DecisionTree(self.criterion, self.max_depth).split(X,y,i))
                max = -9999999999
                max1 = 0 
                max2 = 0 
                for i in range(len(s)):
                    for j in range(len(s[i])):
                        if s[i][j]>max:
                            max1 = i
                            max2 = j
                div_node = h[max1]
                div_ind = max2
                X_sort = X.sort_values(div_node)
                X_sort = X_sort[div_node]
                div_val = (X_sort.iloc[div_ind]+X_sort.iloc[div_ind+1])/2
                Xd1 = X.loc[X[div_node] <= div_val]
                Xd2 = X.loc[X[div_node] > div_val]
                Xd1.drop([div_node],axis=1,inplace=True)
                Xd2.drop([div_node],axis=1,inplace=True)                
                self.node[div_node]=[]
                self.node[div_node].append(div_val)
                self.node[div_node].append(DecisionTree(self.criterion, self.max_depth-1).fit(Xd1, y.loc[X[div_node] <= div_val].copy()))
                self.node[div_node].append(div_val)
                self.node[div_node].append(DecisionTree(self.criterion, self.max_depth-1).fit(Xd2, y.loc[X[div_node] > div_val].copy()))  

        else:
            
            if self.max_depth==0 or len(np.unique(y))==1 or len(y) == 1 or len(X.columns)==1:
                return {"output":sum(y)/y.size}


            if y.dtype != float:
                if self.max_depth!=0:
                    mode = DecisionTree(self.criterion, self.max_depth).getnd(X,y)
                    print(mode)
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
                    s.append(DecisionTree(self.criterion, self.max_depth).split(X,y,i))
                max = -9999999999
                max1 = 0 
                max2 = 0 
                for i in range(len(s)):
                    for j in range(len(s[i])):
                        if s[i][j]>max:
                            max1 = i
                            max2 = j
                div_node = h[max1]
                div_ind = max2
                X_sort = X.sort_values(div_node)
                X_sort = X_sort[div_node]
                div_val = (X_sort.iloc[div_ind]+X_sort.iloc[div_ind+1])/2
                Xd1 = X.loc[X[div_node] <= div_val]
                Xd2 = X.loc[X[div_node] > div_val]
                Xd1.drop([div_node],axis=1,inplace=True)
                Xd2.drop([div_node],axis=1,inplace=True)                
                self.node[div_node]=[]
                self.node[div_node].append(div_val)
                self.node[div_node].append(DecisionTree(self.criterion, self.max_depth-1).fit(Xd1, y.loc[X[div_node] <= div_val].copy()))
                self.node[div_node].append(div_val)
                self.node[div_node].append(DecisionTree(self.criterion, self.max_depth-1).fit(Xd2, y.loc[X[div_node] > div_val].copy()))         

        return self.node
                


    def predict(self, X):
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y=[]
        if "float" not in str(type(X.iloc[0,0])):
            for i in range(len(X)):
                d = self.node
                w = 0
                q = list(d.keys())[0]
                while(isinstance(d[q],list)):
                    w = d[q].index(X.iloc[i,X.columns.get_loc(q)])
                    d = d[q][w+1]
                    q = list(d.keys())[0]
                if q=='output':
                    y.append(d[q])
        else:
            for i in range(len(X)):
                d = self.node
                w = 0
                q = list(d.keys())[0]
                while(isinstance(d[q],list)):
                    if X.iloc[i,X.columns.get_loc(q)]<=d[q][0]:
                        w=0
                        d = d[q][w+1]
                        q = list(d.keys())[0]
                    else:
                        w=2
                        d = d[q][w+1]
                        q = list(d.keys())[0]
                if q=='output':
                    y.append(d[q])
        return pd.Series(y)



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
        
