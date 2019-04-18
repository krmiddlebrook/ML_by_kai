import numpy as np
import numpy.random
import matplotlib.pyplot as plt


def plotLine(a,xMin,xMax,yMin,yMax):
    
    xVals = np.linspace(xMin,xMax,100)
    yVals = (-a[0]*xVals - a[2])/a[1]
    
    idxs = \
        np.where((yVals >= yMin) & (yVals <= yMax))
    
    plt.plot(xVals[idxs],yVals[idxs])


def perceptron(Xpos,Xneg,t):
    numEpochs = 50000 
    a = np.random.randn(3)
    X = np.concatenate((Xpos, Xneg),axis = 0)
    xPosSize = Xpos.shape[0]
    N = X.shape[0]
    xMin = min(Xneg[:,0])
    xMax = max(Xpos[:, 0])
    yMin = min(Xneg[:, 1])
    yMax = max(Xpos[:, 1])
    
    for epoch in range(numEpochs):
        for i in np.random.permutation(N):
            xi = X[i, :]
            if i < xPosSize:
                if np.vdot(a,xi) < 0:
                    a = a + t*xi
            else:
                if np.vdot(a, xi) > 0:
                    a = a - t*xi
        if epoch % 500 == 0:
            print('epoch:' + str(epoch))
            plt.gcf().clear()
            plt.scatter(Xpos[:, 0],Xpos[:,1])
            plt.scatter(Xneg[:,0],Xneg[:,1])
            plotLine(a, xMin, xMax, yMin, yMax)
            plt.axis('equal')
            plt.pause(0.5)
    return a
    
numPos = 100
numNeg = 100

np.random.seed(14)
muPos = [1.0,1.0]
covPos = np.array([[1.0,0.0],[0.0,1.0]])

muNeg = [-1.0,-1.0]
covNeg = np.array([[1.0,0.0],[0.0,1.0]])

Xpos = np.ones((numPos,3))
for i in range(numPos):
    Xpos[i,0:2] = \
        np.random.multivariate_normal(muPos,covPos)
        
Xneg = np.ones((numNeg,3))
for i in range(numNeg):
    Xneg[i,0:2] = \
        np.random.multivariate_normal(muNeg,covNeg)
        

t = .000001
a = perceptron(Xpos,Xneg,t)

#################################################
######## Below are some useful code examples #####
#################################################

# plt.gcf().clear() # clears the current figure
a = np.random.randn(3) 
prm = np.random.permutation(5) # Creates a random permutation of 0, 1, 2, 3, 4 
x5 = Xpos[5,:] # Extracts the fifth row of the array Xpos
dotProd = np.vdot(a,x5) # Computes the dot product of a with x5

if 100 % 20 == 0:
    print("100 is a multiple of 20.")
    
s = Xpos.shape
numPosExamples = s[0]

 









 