
# This code solves a least squares problem with synthetic data.
import numpy as np
import matplotlib.pyplot as plt

numRows = 1000
numCols = 100

# Create feature vectors, stored as rows of matrix X
X = np.random.randn(numRows,numCols)

# Now create the observation values Y = X*beta + noise
betaTrue = np.random.randn(numCols,1) 
sigma = 5 # sigma controls the amount of Gaussian noise.
noise = sigma*np.random.randn(numRows,1)
Y = X@betaTrue + noise

# Now set up the normal equations and solve for beta
XTransX = X.T@X
XTransY = X.T@Y
betaEst = np.linalg.solve(XTransX,XTransY)

# Now check how well betaEst agrees with betaTrue
diff = np.linalg.norm(betaTrue - betaEst)

plt.figure()
plt.plot(betaTrue)
plt.plot(betaEst) 
# Try different values of sigma to see how
# adding more or less noise affects the result.

# Assignment: minimize F(beta) = (1/2)||X*beta - Y||^2 using gradient descent,
# and check that the estimate of beta computed using gradient descent
# agrees with betaEst. Plot the objective function value F(beta) vs. iteration.
# Do you observe that the objective function value is decreasing?

def eval_f(X,B,Y):
    
    sum_f = 0
    n = X.shape[0]
    
    for i in range(n):
        diff = (np.vdot(X[i,:],B) - Y[i])**2
        sum_f += diff
        
    return (1/2)*sum_f

def grad_f(X, B, Y):
     
    grad = X.T@(X@B-Y)
    return grad


def gradDescent(X, Y, t):
    
    maxIter = 1000
    n = X.shape[1]
    beta = np.zeros((n,1))
    costs = np.zeros(maxIter)

    for i in range(maxIter):
        grad = grad_f(X, beta, Y)
#        beta = np.subtract(beta, (t*grad))
        beta = beta - t*grad
        costs[i] = eval_f(X, beta, Y)
        
    return beta, costs


# plots
t_vals = [0.001, 0.0001, 0.0001]
for i in range(len(t_vals)):    
    betas, costs = gradDescent(X, Y, t_vals[i])   
    plt.semilogy(costs[1:100])

        
        
    
    
        


