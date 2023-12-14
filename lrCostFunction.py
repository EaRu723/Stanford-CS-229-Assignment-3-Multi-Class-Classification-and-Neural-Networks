import numpy as np
from sigmoid import *


def lr_cost_function(theta, X, y, lmd):
    # Number of training examples
    m = y.size

    # Initialize the cost and gradients to be zero
    cost = 0
    grad = np.zeros(theta.shape)

    # Compute the hypothesis for all examples
    # np.dot(X, theta) computes the hypothesis for every feature and parameter respectively
    # use the sigmoid function to map these predictions between zero and 1
    hypothesis = sigmoid(np.dot(X, theta))

    # Exclude the first theta term (bias term) from regularizatoin
    reg_theta = theta[1:]

    # Computes the cost with regularization
    # The first term is classic logistic regression cost
    # The second term is the regularization term
    term1 = -y * np.log(hypothesis)
    term2 = np.subtract(1,y) * np.log(np.subtract(1, hypothesis))
    cost = np.sum(term1 - term2) / m + (lmd / (2 * m)) * np.sum(reg_theta ** 2)

    # Computes the difference between the predicted and actual value
    error = np.subtract(hypothesis, y)

    # Calculates the gradient of the cost function with respect to every theta
    grad = np.dot(X.T, error) / m
    
    # Apply regularizaton to the gradient (exclusing the bias term)
    grad[1:] = grad[1:] + reg_theta * (lmd / m)



    return cost, grad
