import numpy as np
from random import shuffle
from past.builtins import xrange
import math

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  fx = X.dot(W)
  fx -= np.max(fx)
  for i in range(X.shape[0]):
    tsum = np.exp(fx[i]).sum()
    loss += -fx[i, y[i]] + math.log(tsum)
    q = np.zeros( (W.shape[1],) )
    for j in range(W.shape[1]):
      q[j] = math.exp(fx[i, j]) / tsum
      for k in range(X.shape[1]):
        dW[k][j] += X[i][k] * (q[j] - (y[i] == j)) / X.shape[0]
  
  dW += 2 * reg * W
  loss /= X.shape[0]
  loss += reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  y_onehot = np.zeros((X.shape[0], W.shape[1]))
  y_onehot[np.arange(X.shape[0]), y] = 1

  fx = X.dot(W)
  fx -= np.max(fx)
  tsum = np.exp(fx).sum(axis=1)
  loss = np.sum( -fx[np.arange(X.shape[0]), y] + np.log(tsum) )
  loss = loss / X.shape[0] + reg * np.sum(W * W)

  tsum = np.array([tsum, ] * W.shape[1]).T
  q = np.exp(fx) / tsum
  dW = X.T.dot(q - y_onehot)
  dW = dW / X.shape[0] + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

