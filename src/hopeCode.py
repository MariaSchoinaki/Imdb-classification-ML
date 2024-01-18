import pandas as pd
import tensorflow as tf
import numpy as np
from math import log 
import math

class HopeLogisticRegression:

    def __init__(self,learning_rate=0.01,max_iter=10):
        self.learning_rate  = learning_rate
        self.max_iter = max_iter
        self.bernulies = []
        self.weights = []
        self. bPredictions=[]

    def logisticFunction(self,t):
        return 1/(1+np.exp(-t.astype(np.float32)))

    def positiveProbapility(self,x,w):
        return 1/(1+np.exp(np.dot(-w,x)))
    
    def negativeProbability(self,x,w):
        return 1-self.positiveProbapility(x,w)
    

    def cost_func(self,X,y_actual,w):
       prob = self.positiveProbapility(X, w)
       prob2 = self.negativeProbability(X,w)
       epsilon = 1e-15
       return np.mean(y_actual*np.log(prob+epsilon)+(1-y_actual)*np.log(prob2+epsilon))
    
    def fit(self, X, y):
        self.weights=X.shape[0]
        costs = 0.0
        for x in range(self.max_iter):
            t=np.dot(X,self.weights)
            y_predicted = self.logisticFunction(t)

            ascent = np.mean((y[x]-y_predicted[0][x])*X.T[0][x])

            self.weights = self.weights + self.learning_rate*ascent

           
            costs  =  self.cost_func(X[x][0],y[x],self.weights)

            self.bernulies.append(costs)

    def predict(self, X ):
        


        bound = 0.5
        t = np.dot(X,self.weights)
        R = self.logisticFunction(t)
        for x in R:
            if x[1] > bound:
                self.bPredictions.append(1)
            else:
               self.bPredictions.append(0)
        

    def TruePositive(self,Truth,model_truth):
        count = 0
        for x, y in zip(Truth,model_truth):
            if x and y ==1:
                count+=1
        return count
    
    def FalsePositive(self,Truth,model_truth):
        count = 0
        for x, y in zip(Truth,model_truth):
            if x == 0 and y ==1:
                count+=1
        return count
    
    def TrueNegative(sself,Truth,model_truth):
        count = 0
        for x, y in zip(Truth,model_truth):
            if x == 0 and y ==0:
                count+=1
        return count
    
    def FalseNegative(self,Truth,model_truth):
        count = 0
        for x, y in zip(Truth,model_truth):
            if x == 1 and y ==0:
                count+=1
        return count
    

    def presicion(self,Truth):
        tp = self.TruePositive(Truth,self.bPredictions)
        fp = self.FalsePositive(Truth,self.bPredictions)

        return tp/(tp+fp)
    
    def recall(self,Truth):
        tp = self.TruePositive(Truth,self.bPredictions) 
        fn = self.FalseNegative(Truth,self.bPredictions)

        return tp/(tp+fn)

    def f1Score(self,Truth):
        p = self.presicion(Truth,self.bPredictions)
        r = self.recall(Truth, self.bPredictions)
        f1score = 2 * p * r/ (p + r) 
        return f1score

        

            

    