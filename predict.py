import numpy as np
from sigmoid import *

def predict(theta1, theta2, x):
    # Get the number of examples (m)
    m = x.shape[0]
    # Get the number of outpu labels from the shape of theta2
    num_labels = theta2.shape[0]

    # Initialize prediction array with zeros
    p = np.zeros(m)

    # Add a column of 1s to the input features x as bias unit
    x = np.c_[np.ones(m), x]

    # First layer computation:
    # Activate first layer by applying sigmoid to dot prodic of x and theta 1 transpose
    h1 = sigmoid(np.dot(x, theta1.T))

    # Add a column of 1s to the output of the first layer as bias unit for the next layer
    h1 = np.c_[np.ones(h1.shape[0]), h1]

    # Second layer (output layer) computation:
    # Activate the second layer by appying sigmoid to the dot product of h1 and theta 2 transpose
    h2 = sigmoid(np.dot(h1, theta2.T))

    # Determine the predicted class for each example
    # np argmax determins the index for the most probable class
    # Add + 1 to account for Python's 0-based indexing.
    p = np.argmax(h2, axis =1) + 1

    return p


