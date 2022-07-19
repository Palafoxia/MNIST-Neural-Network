import numpy as np
from functions import *

# FORWARD PROPAGATION
def forward(X, params):
    # Inputs:
      # X -- 784 x N array 
      # params
        # W1 -- 256 x 784 matrix
        # b1 -- 256 x 1 vector
        # W2 -- 10 x 256 matrix
        # b2 -- 10 x 1 vector 
      
    # Outputs:
      # probs -- 10 x N output

    Z1 = np.matmul(params[0], X) + params[1][:, np.newaxis] # W1X + b1
    Y1 = sigmoid(Z1) # Output of layer 1
    Z2 = np.matmul(params[2], Y1) + params[3][:, np.newaxis] # W2Y1 + b2
    probs = stable_softmax(Z2) # Final output

    # Put all intermediate values in array and return
    intermediate = np.array([X, Z1, Y1, Z2], dtype=object)
    return probs, intermediate

# BACKPROPAGATION
def backward(Y_true, probs, intermediate, params):
    # Inputs: 
      # Y_true -- 1 x N true labels
      # probs -- 10 x N output of the last layer
      # intermediate -- X, Z1, Y1, Z2 
      # params -- W1, b1, W2, b2 
    
    # Outputs: 
      # grads -- [grad_W1, grad_b1, grad_W2, grad_b2]

    # Num Samples
    N = np.size(Y_true)

    # LAYER 2 (l = L)
    delta2 = np.zeros((10, N))
    Y1 = intermediate[2]

    # Compute delta2
    for i in range(N):
      oneHot = np.zeros(10)
      oneHot[Y_true[i].astype(int)] = 1
      delta2[:, i] = probs[:, i] - oneHot

    # Compute b2GradientAverage/W2GradientAverage
    b2GradientAverage = np.zeros((10))
    for j in range(10):
      b2GradientAverage[j] = np.average(delta2[j, :])
    W2GradientAverage = (1/N) * np.matmul(delta2, Y1.T) # (10, 256)

    # LAYER 1 (l < L)
    W2 = params[2]
    Z1 = intermediate[1]
    
    # Compute delta1
    delta1 = np.matmul(W2.T, delta2) * (sigmoid(Z1) * (1 - sigmoid(Z1))) # (256, 10000)

    # Compute b1GradientAverage
    b1GradientAverage = np.zeros((256))
    for j in range(256):
      b1GradientAverage[j] = np.average(delta1[j, :])

    # Compute W1GradientAverage
    X = intermediate[0]
    W1GradientAverage = (1/N) * np.matmul(delta1, X.T) # (256, 784)

    # Put weight/bias gradients in array and return
    grads = np.array([W1GradientAverage, b1GradientAverage, W2GradientAverage, b2GradientAverage], dtype=object)
    return grads

# GRADIENT DESCENT OPTIMIZER
def GD(params, grads, learning_rate):
    # New params = old params - (learning rate (α) * gradient of Loss computed at old params)
    # params -- W1, b1, W2, b2 
    params[0] = params[0] - (learning_rate * grads[0]) #W1 - αLossW1 (256, 784)
    params[1] = params[1] - (learning_rate * grads[1]) #b1 - αLossb1 (256, )
    params[2] = params[2] - (learning_rate * grads[2]) #W2 - αLossW2 (1, 256)
    params[3] = params[3] - (learning_rate * grads[3]) #b2 - αLossb2 (1, )
    return params