import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
	"""
	Structured SVM loss function, naive implementation (with loops).

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
	dW = np.zeros(W.shape) # initialize the gradient as zero

	# compute the loss and the gradient
	num_classes = W.shape[1]
	num_train = X.shape[0]
	loss = 0.0
	for i in xrange(num_train):
		scores = X[i].dot(W)
		correct_class_score = scores[y[i]]
		for j in xrange(num_classes):
			if j == y[i]:
				continue
			margin = scores[j] - correct_class_score + 1 # note delta = 1
			if margin > 0:
				loss += margin

	# Right now the loss is a sum over all training examples, but we want it
	# to be an average instead so we divide by num_train.
	loss /= num_train

	# Add regularization to the loss.
	loss += reg * np.sum(W * W)

	#############################################################################
	# TODO:                                                                     #
	# Compute the gradient of the loss function and store it dW.                #
	# Rather that first computing the loss and then computing the derivative,   #
	# it may be simpler to compute the derivative at the same time that the     #
	# loss is being computed. As a result you may need to modify some of the    #
	# code above to compute the gradient.                                       #
	#############################################################################

	new_loss = np.zeros(W.shape)
	for i in xrange(num_train):
		scores = X[i].dot(W)
		correct_class_score = scores[y[i]]
		for d in xrange(W.shape[0]):
			for j in xrange(num_classes):
				if j == y[i]:
					for k in xrange(num_classes):
						if k == y[i]:
							continue
						correct_class_score += X[i][d] * 0.0001 

						new_margin = max(0, scores[k] - correct_class_score + 1)
						correct_class_score -= X[i][d] * 0.0001 
						margin = max(0, scores[k] - correct_class_score + 1)
						new_loss[d][j] += (new_margin - margin) / num_train

					continue

				
				new_margin = max(0, scores[j] + X[i][d] * 0.0001 - correct_class_score + 1)
				margin = max(0, scores[j] - correct_class_score + 1)
				new_loss[d][j] += (new_margin - margin) / num_train


	for i in xrange(W.shape[0]):
		for j in xrange(W.shape[1]):
			W[i][j] += 0.0001
			new_loss[i][j] += reg * np.sum(W * W)
			W[i][j] -= 0.0001
			new_loss[i][j] -= reg * np.sum(W * W)

			dW[i][j] = new_loss[i][j] / 0.0001

	return loss, dW


def svm_loss_vectorized(W, X, y, reg):
	"""
	Structured SVM loss function, vectorized implementation.

	Inputs and outputs are the same as svm_loss_naive.
	"""
	loss = 0.0
	num_classes = W.shape[1]
	num_train = X.shape[0]
	dW = np.zeros(W.shape) # initialize the gradient as zero

	#############################################################################
	# TODO:                                                                     #
	# Implement a vectorized version of the structured SVM loss, storing the    #
	# result in loss.                                                           #
	#############################################################################

	scores = X.dot(W)
	correct_scores = [scores[i][y[i]] - 1 for i in range(num_train)]
	correct_class_scores = np.array([correct_scores,] * num_classes).T
	diag = np.zeros((num_train, num_classes))
	diag[np.arange(num_train), y] = 1
	margins = scores - correct_class_scores - diag
	margins[margins < 0] = 0

	loss = np.sum(margins)
	loss /= num_train
	loss += reg * np.sum(W * W)

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################


	#############################################################################
	# TODO:                                                                     #
	# Implement a vectorized version of the gradient for the structured SVM     #
	# loss, storing the result in dW.                                           #
	#                                                                           #
	# Hint: Instead of computing the gradient from scratch, it may be easier    #
	# to reuse some of the intermediate values that you used to compute the     #
	# loss.                                                                     #
	#############################################################################

	new_loss = np.zeros(W.shape)
	for d in xrange(W.shape[0]):

		for j in xrange(num_classes):
			scores[:, j] += X[:, d] * 0.00001 * (y != j)
			new_margins = scores - correct_class_scores - diag
			new_margins[new_margins < 0] = 0
			scores[:, j] -= X[:, d] * 0.00001 * (y != j)
			new_loss[d, j] = ( new_margins - margins ).sum() / num_train

		new_correct = correct_scores + X[:, d] * 0.00001
		new_class_corrects = np.array([new_correct,] * num_classes).T
		scores[np.arange(num_train), y] += X[:, d] * 0.00001
		new_margins = scores - new_class_corrects - diag
		scores[np.arange(num_train), y] -= X[:, d] * 0.00001
		new_margins[new_margins < 0] = 0
		a = np.sum(( new_margins - margins ), axis = 1) / num_train

		for j in xrange(0, num_classes):
			new_loss[d, j] += np.sum(a[ np.where(y==j) ])

	dW = reg * (W + 0.00001) * (W + 0.00001) - reg * W * W

	dW = (dW + new_loss) / 0.00001

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################

	return loss, dW
