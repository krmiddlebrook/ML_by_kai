'''
author: Kai Middlebrook

Code to recover the weight vector x, given b and A and prior knowledge that x
is spare, by solving a Lasso problem.
'''

import numpy as np
import matplotlib.pyplot as plt
#%%
numRows = 40
numCols = 200
np.random.seed(6)
A = np.random.randn(numRows,numCols)
idxs = np.random.permutation(numCols)

xTrue = np.zeros((numCols,1))
xTrue[idxs[0:2]] = 5.0
xTrue[idxs[2:4]] = -10.0

noise = .1*np.random.randn(numRows,1)
b = A@xTrue + noise

plt.figure()
plt.stem(b)
plt.title('b (How can we recover xTrue from b?)')

plt.figure()
plt.stem(xTrue)
plt.title('xTrue (most components are zero)')

#%% Now recover xTrue, given b and A, by solving a Lasso problem.
# A and b are known, but xTrue is unknown, and the goal is to find xTrue.
# You'll need to minimize (1/2)|| Ax - b ||^2 + gamma*||x||_1
# using the proximal gradient method. You can set gamma = 10.0.
# The optimal value of x might not agree perfectly with xTrue,
# but the sparsity pattern (that is, the location of the non-zero entries)
# should be correct.

def prox1norm(xhat, lam):
    
    numRows, numCols = xhat.shape
    
    x = np.zeros((numRows, numCols))
    for i in range(numRows):
        
        if xhat[i] > 0:        
            x[i] = np.maximum(xhat[i]-lam, 0)
        else:
            x[i] = np.minimum(xhat[i]+lam, 0)
    
    return x
          

def lasso_proxGrad(A, b, t, gamma):
    
    num_epochs = 5000
    numRows, numCols = A.shape
    costs = np.zeros(num_epochs)
    
    xhat = np.random.rand(numCols,1)
    
    for ep in range(num_epochs):
        
        grad = A.T@(A@xhat-b)
        xhat = prox1norm(xhat-t*grad, t*gamma)
        costs[ep] = 0.5*np.linalg.norm(A@xhat-b)**2 + gamma*np.sum(np.abs(xhat))

    return xhat, costs
#%%

t = 0.0001
gamma = 10
x, costs = lasso_proxGrad(A, b, t=t, gamma=gamma)
#%%
plt.figure()
plt.semilogy(costs)
plt.title(label='log(loss) vs iteration')

plt.figure()
plt.stem(x)
plt.stem(xTrue, markerfmt='C1o')
plt.title(label='X (orange) and our recovered estimation of X (blue)')