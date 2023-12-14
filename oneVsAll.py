import scipy.optimize as opt
import lrCostFunction as lCF
from sigmoid import *


def one_vs_all(X, y, num_labels, lmd):
    # Get the number of training examples (m) and the number of features (n)
    (m, n) = X.shape

    # Initialize a matrix to store the parameters (thetas) for each class/label (digit)
    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the feature matrix (X)
    X = np.c_[np.ones(m), X]

    # Loop through each class/label (digit)
    for i in range(num_labels):
        # Initialize the parameters for the current class to 0
        initial_theta = np.zeros((n + 1, 1))
        # Re-map the label 0 to correspond to 10
        iclass = i if i else 10
        # Create a binary vector to show whether each training example belongs to class i
        y_i = np.array([1 if x == iclass else 0 for x in y])
        print('Optimizing for handwritten number {}...'.format(i))

        # Call the cost function for the current class
        def cost_func(t):
            return lCF.lr_cost_function(t, X, y_i, lmd)[0]
        # Call the gradient function for the current class
        def grad_func(t):
            return lCF.lr_cost_function(t, X, y_i, lmd)[1]
        
        # Optomize parameters using conjugate gradient optimization
        # fmin_cg finds the parameters that minimize the cost function
        # max iter controls the number of iterations for the optimization
        # disp controls if optimizatoin outputs the convergence message
        theta, *unused = opt.fmin_cg(cost_func, fprime = grad_func, x0 = initial_theta, maxiter = 100, disp = False, full_output = True)

        print('Done')

        # Store the optimized parameters in the all_theta matrix
        all_theta[i] = theta

    return all_theta
