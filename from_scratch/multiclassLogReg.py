#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:38:13 2019

@author: kaimiddlebrook
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#%%

                

def eval_f(X, B, Y):
    # objective function F(B) for multi-class logistic regression
    
    numExamples, numFeatures = X.shape
    numExamples, numClasses = Y.shape
    
    cost = 0
    
    for i in range(numExamples):
        
        xi = X[i,:] # ith training example dims = (1, numFeatures)
        yi = Y[i,:] # ith y vector dims = (1, numClasses)            
        dotProds = xi@B  # the dot products for xi and every B1,...,Bk; dims = (1, numClasses)
        terms = np.exp(dotProds) # e^xiB
        terms_sum = np.sum(terms)
        p = terms/terms_sum # e^xiBk/sum(e^xiB); p is a list of probabilities
        
        k = np.argmax(yi) # k is the index in the yi vector that equals 1
        cost += yi[k] * np.log(p[k])
            
    return -cost
 
def grad_f(X, B, Y):
    # calculates the gradients for the beta matrix
    
    numExamples, numFeatures = X.shape
    numExamples, numClasses = Y.shape
    grad = np.zeros((numFeatures, numClasses))
    
    for i in range(numExamples):
        
        xi = X[i,:]
        yi = Y[i]
        dotProds = xi@B  # the dot products for xi and every B1,...,Bk; dims = (1, numClasses)
        terms = np.exp(dotProds) # e^xiB
        terms_sum = np.sum(terms)
        p = terms/terms_sum # e^xiBk/sum(e^xiB); p is a list of probabilities
        
        for k in range(numClasses):
            
            grad[:,k] += (p[k] - yi[k])*xi
    
    return grad
        
def grad_f_sgd(xi, B, yi):
    '''
    calculates the gradient for the ith example
    inputs: 
        (1) xi: the ith training example from the feature matrix X
        (2) B: a matrix containing the weights for each class B_0, ..., B_k dims = (d x k) where k = the number of classes
        (3) yi: the ith onehot encoded vector containing the true class for the ith example
    return:
        (1) grad: 
    '''
    numFeatures = xi.shape[0]
    numClasses = yi.shape[0]
    grad = np.zeros((numFeatures, numClasses))
    
    dotProds = xi@B  # the dot products for xi and every B1,...,Bk; dims = (1, numClasses)
    terms = np.exp(dotProds) # e^xiB
    terms_sum = np.sum(terms)
    p = terms/terms_sum # e^xiBk/sum(e^xiB); p is a list of probabilities
    
    for k in range(numClasses):
        
        grad[:,k] = (p[k] - yi[k])*xi
        
        
    return grad

def multiclass_logReg_GD(X, Y, t):
    '''
    fits a model to the data using multi-class logistic regression using gradient descent
    inputs:
        (1) X:  a matrix of feature vectors dims = (n x d+1) where n = number of examples and d = number of features 
        (2) Y: a onehot encoded vector with the true classes for every example
        (3) t: the learning rate
    return:
        (1) beta: a matrix containing the minimized Beta values for each class dims = (d x k) where k = number of classes
        (2) costs: the objective function loss after each epoch
    '''
    
    maxIter = 5000
    showTrigger = 5
    numExamples, numFeatures = X.shape
    numExamples, numClasses = Y.shape
    
    beta = np.random.randn(numFeatures,numClasses)
    costs = np.zeros(maxIter)
    
    for idx in range(maxIter):
        
        grad = grad_f(X, beta, Y)
        beta = beta - t*grad
        
        costs[idx] = eval_f(X, beta, Y)
        
        if idx % showTrigger == 0:
            print("Iteration: " + str(idx) + " Cost: " + str(costs[idx]))
              
    return beta, costs
        
 
def multiclass_logReg_SGD(X, Y, t, numEpochs):
    '''
    fits a model to the data using multi-class logistic regression and stochastic gradient descent
    inputs:
        (1) X:  a matrix of feature vectors dims = (n x d+1) where n = number of examples and d = number of features 
        (2) Y: a onehot encoded vector with the true classes for every example
        (3) t: the learning rate
        (4) numEpochs: the number of epochs to run before returning beta
    return:
        (1) beta: a matrix containing the minimized Beta values for each class dims = (d x k) where k = number of classes
        (2) costs: the objective function loss after each epoch
    '''
    showTrigger = 5
    numExamples, numFeatures = X.shape
    numExamples, numClasses = Y.shape
    
    beta = np.random.randn(numFeatures,numClasses)
    costs = np.zeros(numEpochs)
    
    for ep in range(numEpochs):
        
        for i in np.random.permutation(numExamples):
            
            grad = grad_f_sgd(X[i,:], beta, Y[i,:])
            beta = beta - t*grad
            
        costs[ep] = eval_f(X, beta, Y)
        
        if ep % showTrigger == 0:
            print("Epoch: " + str(ep) + " Cost: " + str(costs[ep]))
        if showTrigger <= 10 and numEpochs <= 20 and ep % showTrigger != 0:
            print("Epoch: " + str(ep) + " Cost: " + str(costs[ep]))
                
            
    return beta, costs

def multiclass_predictions(X, beta):
    '''
    compute predictions after training a model using mutliclass logistic regression
    inputs:
        (1) X: a matrix of feature vectors dims = (n x d+1) where n = number of examples and d = number of features 
        (2) beta: a matrix containing the weights for each class B_0, ..., B_k dims = (d x k) where k = the number of classes
    return:
        (1) predictions: a vector containing the predictions for every example in X dims = (n x 1)
    '''
    numExamples, numFeatures = X.shape
    numFeatures, numClasses = beta.shape
    predictions = np.zeros(numExamples)
    
    for i in range(numExamples):
        
        xi = X[i,:]
        dotProds = xi@beta
        terms = np.exp(dotProds)
        p = terms/np.sum(terms)
        predicted_k = np.argmax(p)
        predictions[i] = predicted_k
        
    return predictions
   
def classification_acc(Y, predictions):
    '''
    calculates the classification accuracy of our multiclass logistic regression model
    inputs:
        (1) Y: a onehot encoded vector with the true classes for every example
        (2) predictions: a vector containing the predictions for every example
    return:
        (1) the overall classification accuracy
    '''
    numExamples, numClasses = Y.shape
    numCorrect = 0
    
    for i in range(numExamples):
        
        true_class = np.argmax(Y[i])
        if predictions[i] == true_class:
            numCorrect += 1
    classification_accuracy = round(numCorrect / numExamples, 4)
    return classification_accuracy
    
        
        
def plotObjFuncVsIter(objFuncCosts):
    
    plt.figure()
    plt.semilogy(objFuncCosts)
    plt.title("Stochastic Gradient descent cost vs epoch")  
               
#%%


    