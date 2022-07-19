import numpy as np

# SIGMOID ACTIVATION FUNCTION
def sigmoid(Z):
    # Input: Z numpy.ndarray
    Y = 1/(1 + np.exp(-Z))
    return Y

# STABLE SOFTMAX ACTIVATION FUNCTION
def stable_softmax(probs): 
    # probs -- K x N numpy.ndarray, K is the number of classes, N is the number of samples
    probs = np.exp(probs - np.max(probs)) / np.sum(np.exp(probs - np.max(probs)), axis = 0)
    return probs

# CROSS ENTROPY LOSS FUNCTION FOR MULTIPLE CLASSES
def MultiClassCrossEntropyLoss(Y_true, probs):
  # probs -- K x N array
  # Y_true -- 1 x N array 
  # loss -- sum Loss_i over N samples 

  N = Y_true.shape[0] # N Samples
  p = stable_softmax(probs)
  log_likelihood = -np.log(p[Y_true.astype(int), range(N)])
  loss = np.sum(log_likelihood)/N
  return loss

# Truncate float to n decimal places
def truncate(f, n):
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])