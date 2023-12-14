import numpy as np


def predict_one_vs_all(all_theta, X):
    # Get the number of examples
    m = X.shape[0]
    # Get the number of labels
    num_labels = all_theta.shape[0]

    # Initialize the prediction array with zeros
    p = np.zeros(m)

    # Add a column of ones to input matrix 'X' for the bias/intercept term
    X = np.c_[np.ones(m), X]

    # Multiply all_theta with the transpose of X to get the class scores for each example
    # all_theta (num_labels x n+1) x X.T (n+1 x m) = (num_labels x m)
    # Each element [i,j] in the result represents the score for class i for example j
    result = np.dot(all_theta, X.T)

    # Adjusts the indices of the classes to account for the fact that python is 0-indexed
    result = np.roll(result, -1, axis=0)

    # Adds a row of zeros at the beginning of the result array
    # Aligns class indices if the classes are labeled 1 through 10
    result = np.vstack([np.zeros(m), result])

    # Use np.argmax to find the index of the maximum score in each column
    # This gives the most probable class for each example in X
    # Finds the index of the maximum in each column
    p = np.argmax(result, axis=0)

    

    return p
