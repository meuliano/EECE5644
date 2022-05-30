# Intro to Machine Learning and Pattern Recognition - EECE5644
# Homework 1 - Problem 2
# Author: Matthew Euliano
import matplotlib.pyplot as plt #General Plotting
import numpy as np
from scipy.stats import multivariate_normal


# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(4)

N = 10000 # samples

def main():
    priors = np.array([.2,.25,.25,.3]) # Given Priors

    # ARBITRARILY CHOSEN mean and covariance matrix
    # equally spaced means along a line
    mu =np.array([[0, 0],[3, 0],[6, 0],[9, 0]])

    # scaled versions of the identity matrix with overlap
    sigma =[[[2, 0],[0, 2]],
            [[4, 0],[0, 4]],
            [[6, 0],[0, 6]],
            [[8, 0],[0, 8]]]

    n = mu.shape[1] # sample dimensions (2)
    C = len(priors) # Classes (4)
    L = np.array(range(C)) # labels [0 1 2 3]
    Y = np.array(range(C))  # 0-(C-1)

    # set thresholds: results in [0.0, 0.2, 0.45, 0.7, 1.0]
    thresholds = np.cumsum(priors)
    thresholds = np.insert(thresholds,0,0)

    X = np.zeros([N,n]) # create empty matrix for samples of size 10,000 x 3
    labels = np.zeros(N) # create empty labels array of length 10,000

    # Create a 10000-length vector of uniformly distributed random variables in [0,1)
    # Will be used to get an estimate of number of features belonging to each label
    u = np.random.rand(N)

    # Plot for original data and their true labels
    fig = plt.figure(figsize=(10, 10))
    marker_shapes = '....'
    marker_colors = 'rbgy' 

    for i in range(C):

        # Find indices of u that meet each prior
        indices = np.argwhere((thresholds[i] <= u) & (u <= thresholds[i+1]))[:, 0]

        # Get the number of indices in each component - should be ~ 6500 and 3500
        Nl = len(indices)
        
        # set label vector based on above - will be vector of class labels [0 0 1 0 1 .. to 9999] in this case
        # for more classes, this can be [1 2 2 0 1 3 1 3 0 ...]
        labels[indices] = i * np.ones(Nl)
        
        # for each valid index...
        X[indices, :] = multivariate_normal.rvs(mu[i], sigma[i], Nl)
        plt.plot(X[labels==i, 0], X[labels==i, 1], marker_shapes[i-1] + marker_colors[i-1], label="True Class {}".format(i))

    # Plot the original data and their true labels
    plt.legend()
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title("Generated Original Data Samples")
    plt.tight_layout()
    plt.show()

    # Generate data distributions and store in true class labels
    Nl = np.array([sum(labels == i) for i in range(C)])
    print("Num samples: 0 = {}, 1 = {}, 2 = {}, 3 = {}".format(Nl[0],Nl[1],Nl[2],Nl[3]))

    # Lambda Matrix
    lambda_matrix = np.ones((C, C)) - np.identity(C)

    # Calculate class-conditional likelihoods p(x|Y=j) for each label of the N observations
    class_cond_likelihoods = np.array([multivariate_normal.pdf(X, mu[j], sigma[j]) for j in Y])
    class_priors = np.diag(priors)
    class_posteriors = class_priors.dot(class_cond_likelihoods)

    # We want to create the risk matrix of size 3 x N 
    cond_risk = lambda_matrix.dot(class_posteriors)

    # Get the decision for each column in risk_mat
    decisions = np.argmin(cond_risk, axis=0)

    # Plot for decisions vs true labels
    fig = plt.figure(figsize=(12, 10))
    marker_shapes = 'ox+*.' # Accomodates up to C=5
    marker_colors = 'brgmy'

    # Get sample class counts
    sample_class_counts = np.array([sum(labels == j) for j in Y])

    # Confusion matrix
    conf_mat = np.zeros((C, C))
    for i in Y: # Each decision option
        for j in Y: # Each class label
            ind_ij = np.argwhere((decisions==i) & (labels==j))
            conf_mat[i, j] = round(len(ind_ij)/sample_class_counts[j],3) # Average over class sample count
            if i == j:
                # True label = Marker shape; Decision = Marker Color
                marker = marker_shapes[j] + marker_colors[i]
                plt.plot(X[ind_ij, 0], X[ind_ij, 1], 'g'+marker_shapes[j], markersize=6,label="Correct decision {}".format(i))
            else:
                plt.plot(X[ind_ij, 0], X[ind_ij, 1], 'r'+marker_shapes[j], markersize=6,label="Incorrect Decision {} in label {}".format(i,j))
                
    print("Confusion matrix:")
    print(conf_mat)

    print("Minimum Probability of Error:")
    prob_error = 1 - np.diag(conf_mat).dot(sample_class_counts / N)
    print(prob_error)

    plt.legend()
    plt.title("Minimum Probability of Error Classified Sampled Data:  {:.3f}".format(prob_error))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.tight_layout()
    plt.show()

    # %%
    # Part B
    lambda_matrix_b =np.array([[ 0, 1, 2, 3],[1, 0, 1, 2],[2, 1, 0, 1],[3, 2, 1, 0]])

    # We want to create the risk matrix of size 3 x N 
    cond_risk_b = lambda_matrix_b.dot(class_posteriors)

    # Get the decision for each column in risk_mat
    decisions_b = np.argmin(cond_risk_b, axis=0)
    print(decisions_b)

    # Plot for decisions vs true labels
    fig = plt.figure(figsize=(12, 10))
    marker_shapes = 'ox+*.' # Accomodates up to C=5
    marker_colors = 'brgmy'

    # Get sample class counts
    sample_class_counts = np.array([sum(labels == j) for j in Y])

    # Confusion matrix
    conf_mat_b = np.zeros((C, C))
    for i in Y: # Each decision option
        for j in Y: # Each class label
            ind_ij = np.argwhere((decisions_b==i) & (labels==j))
            conf_mat_b[i, j] = round(len(ind_ij)/sample_class_counts[j],3) # Average over class sample count

            if i == j:
                # True label = Marker shape; Decision = Marker Color
                marker = marker_shapes[j] + marker_colors[i]
                plt.plot(X[ind_ij, 0], X[ind_ij, 1], 'g'+marker_shapes[j], markersize=6,label="Correct decision {}".format(i))

            else:
                plt.plot(X[ind_ij, 0], X[ind_ij, 1], 'r'+marker_shapes[j], markersize=6,label="Incorrect Decision {} in label {}".format(i,j))
                
    print("Confusion matrix:")
    print(conf_mat_b)

    print("Minimum Probability of Error:")
    prob_error_b = 1 - np.diag(conf_mat_b).dot(sample_class_counts / N)
    print(prob_error_b)

    plt.legend()
    plt.title("Minimum Probability of Error Classified Sampled Data:  {:.3f}".format(prob_error_b))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()