#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 13:58:44 2019

@author: kaimiddlebrook
"""

import numpy as np 
import numpy.random
import random
import matplotlib.pyplot as plt

def plotline(a, xMin, xMax, yMin, yMax):
    
    xVals = np.linspace(xMin, xMax, 100)
    yVals = -a[0]*xVals - a[2]/a[1]
    
    idxs = np.where((yVals >= yMin) & (yVals <= yMax))   
    
    plt.plot(xVals[idxs], yVals[idxs])

numPos = 100
numNeg = 100
### Create the data ###
muPos = [1.0, 1.0]

covPos = np.array([[1.0, 0.0], [0.0, 1.0]])

muNeg = [-1.0, -1.0]
covNeg = np.array([[1.0, 0.0], [0.0, 1.0]])

Xpos = np.ones((numPos, 3))
for i in range(numPos):
    Xpos[i, 0:2] = np.random.multivariate_normal(muPos, covPos)

Xneg = np.ones((numNeg, 3))
for i in range(numNeg):
    Xneg[i, 0:2] = np.random.multivariate_normal(muNeg, covNeg)
    
X = np.concatenate((Xpos, Xneg), axis = 0)
Ypos = np.zeros((100,1)) + 1
Yneg = np.zeros((100,1))
Y = np.concatenate((Ypos, Yneg), axis = 0)

xMin = -3.0
xMax = 3.0
yMin = -3.0
yMax = 3.0


plt.scatter(Xpos[:,0],Xpos[:, 1])
plt.scatter(Xneg[:,0], Xneg[:,1])
#plotline(a, xMin, xMax, yMin, yMax)
plt.axis('equal')


### logistic regression functions ###

def sigmoid(u):

    expn = np.exp(u)
    return expn / (1 + expn)    
    

def eval_f(X, B, Y):
    
    nrows = X.shape[0]
    
    cost = 0
    
    for i in range(nrows):
        
        xi = X[i,:]
        yi = Y[i]
        
        p = sigmoid(np.vdot(xi, B))
        
        cost += yi*np.log(p) + (1-yi)*np.log(1-p)
        
    return -cost


def grad_f(X, B, Y):
    
    nrows, ncols = X.shape
    
    grad = np.zeros(ncols)
    
    for i in range(nrows):
        xi = X[i,:]
        yi = Y[i]
        
        p = sigmoid(np.vdot(xi, B))
        
        grad = grad + (p-yi)*xi
        
    return grad

def grad_f_SG(xi, B, yi):
    
    p = sigmoid(np.vdot(xi,B))

    grad = (p - yi)*xi
    
    return grad
    
    

def logReg_GD(X, Y, t):
    
    maxIter = 400
    showTrigger = 10
    
    numExamples, numFeatures = X.shape
    
    beta = np.random.randn(numFeatures)
    costs = np.zeros(maxIter)
    
    xMin = min(X[:,0])
    xMax = max(X[:, 0])
    yMin = min(X[:, 1])
    yMax = max(X[:, 1])
    
    for i in range(maxIter):
        
        grad = grad_f(X, beta, Y)
        beta = beta - t*grad
        costs[i] = eval_f(X, beta, Y)
        
        if i % showTrigger == 0:
            print("Iteration: " + str(i) + " Cost: " + str(costs[i]))
            plt.gcf().clear()
            plt.scatter(Xpos[:,0], Xpos[:,1])
            plt.scatter(Xneg[:,0], Xneg[:,1])
            plotline(beta, xMin, xMax, yMin, yMax)
            plt.pause(.05)
            
    return beta, costs



def logReg_SGD(X, Y, t, numEpochs):
    
    showTrigger = 5
    numExamples, numFeatures = X.shape
    
    beta = np.random.randn(numFeatures)
    costs = np.zeros(numEpochs)
    
    xMin = min(X[:,0])
    xMax = max(X[:, 0])
    yMin = min(X[:, 1])
    yMax = max(X[:, 1])
    
    for j in range(numEpochs):
        
        for i in np.random.permutation(numExamples):
            xi = X[i,:]
            yi = Y[i]
            grad = grad_f_SG(xi, beta, yi)
            beta = beta - t*grad
        
        costs[j] = eval_f(X, beta, Y)

        
        if j % showTrigger == 0:
            print("Epoch: " + str(j) + " Cost: " + str(costs[j]))
            plt.gcf().clear()
            plt.scatter(Xpos[:,0], Xpos[:,1])
            plt.scatter(Xneg[:,0], Xneg[:,1])
            plotline(beta, xMin, xMax, yMin, yMax)
            plt.pause(.05)

            
    return beta, costs


gd_beta, gd_costs = logReg_GD(X,Y,0.001)
sgd_beta, sdg_costs = logReg_SGD(X, Y, 0.001, 100)

## semilogy plots of objective function vs iteration (Gradient Descent) or epoch (Stochastic Gradient Descent)
plt.figure()
plt.semilogy(gd_costs)
plt.title("Gradient Descent cost vs iteration")



plt.figure()
plt.semilogy(sdg_costs)
plt.title("Stochastic Gradient Descent cost vs epoch")


    
    
        
    
