# %%
# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sys import float_info # Threshold smallest positive floating value
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix

import pandas as pd

import torch
import torch.nn as nn

# %%
def generate_data_from_gmm(N, pdf_params):
    # Generate Data Function - Returns N x 3 and labels
    # Determine dimensionality from mixture PDF parameters
    n = pdf_params['m'].shape[1]
    # Output samples and labels
    X = np.zeros([N, n])
    labels = np.zeros(N)
    
    # Decide randomly which samples will come from each component
    u = np.random.rand(N)
    thresholds = np.cumsum(pdf_params['priors'])
    thresholds = np.insert(thresholds, 0, 0) # For intervals of classes

    L = np.array(range(1, len(pdf_params['priors'])+1))
    for l in L:
        # Get randomly sampled indices for this component
        indices = np.argwhere((thresholds[l-1] <= u) & (u <= thresholds[l]))[:, 0]
        # No. of samples in this component
        Nl = len(indices)  
        labels[indices] = l * np.ones(Nl) - 1
        if n == 1:
            X[indices, 0] =  norm.rvs(pdf_params['m'][l-1], pdf_params['C'][l-1], Nl)
        else:
            X[indices, :] =  multivariate_normal.rvs(pdf_params['m'][l-1], pdf_params['C'][l-1], Nl)
    
    return X, labels

# %%
def perform_erm_classification(X, Lambda, gmm_params, C):    
    # ERM classification rule (min prob. of error classifier)
    # Conditional likelihoods of each x given each class, shape (C, N)
    class_cond_likelihoods = np.array([multivariate_normal.pdf(X, gmm_params['m'][c], gmm_params['C'][c]) for c in range(C)])

    # Take diag so we have (C, C) shape of priors with prior prob along diagonal
    class_priors = np.diag(gmm_params['priors'])
    # class_priors*likelihood with diagonal matrix creates a matrix of posterior probabilities
    # with each class as a row and N columns for samples, e.g. row 1: [p(y1)p(x1|y1), ..., p(y1)p(xN|y1)]
    class_posteriors = class_priors.dot(class_cond_likelihoods)

    # Conditional risk matrix of size C x N with each class as a row and N columns for samples
    risk_mat = Lambda.dot(class_posteriors)
    
    return np.argmin(risk_mat, axis=0)
# %%
def train_model(model, data, labels, criterion, optimizer, num_epochs=25):
    # Set up training data
    X_train = torch.FloatTensor(data)
    y_train = torch.LongTensor(labels)

    # Optimize the neural network
    for epoch in range(num_epochs):
        # Set grads to zero explicitly before backprop
        optimizer.zero_grad()
        outputs = model(X_train)
        # Criterion computes the cross entropy loss between input and target
        loss = criterion(outputs, y_train)
        # Backward pass to compute the gradients through the network
        loss.backward()
        # GD step update
        optimizer.step()

    return model

# %%
def model_predict(model, data):
    # Set up test data as tensor
    X_test = torch.FloatTensor(data)

    # Evaluate nn on test data and compare to true labels
    predicted_labels = model(X_test)
    # Back to numpy
    predicted_labels = predicted_labels.detach().numpy()
    
    return np.argmax(predicted_labels, 1)

# %%
def mse(y_preds, y_true):
    # Residual error (X * theta) - y
    error = y_preds - y_true
    # Loss function is MSE
    return np.mean(error ** 2)

# %%
def model_order_selection(X_train, y_train, folds, poly_deg):

  C = len(np.unique(y_train))

  cv = KFold(n_splits=folds, shuffle=True)

  # Polynomial degrees ("hyperparameters") to evaluate 
  degs = np.arange(1, poly_deg, 1)
  n_degs = np.max(degs)
  error_prob=np.empty((n_degs,folds))
  all_models = []
  # 
  for deg in degs:
    k = 0

    # split training set for 10-fold cross validation
    # for each of the 10 smaller sets, train model for 1-10 perceptrons
    for fold, (train_indices, valid_indices) in enumerate(cv.split(X_train)):
        # Extract the training and validation sets from the K-fold split
        X_train_k = X_train[train_indices]
        y_train_k = y_train[train_indices]
        X_valid_k = X_train[valid_indices]
        y_valid_k = y_train[valid_indices]

        model = TwoLayerMLP(X_train.shape[1], deg, C)      
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        # The nn.CrossEntropyLoss() loss function automatically performs a log_softmax() to 
        # the output when validating, on top of calculating the negative-log-likelihood using 
        # nn.NLLLoss(), while also being more stable numerically... So don't implement from scratch
        criterion = nn.CrossEntropyLoss()
        trained_model = train_model(model, X_train_k, y_train_k, criterion, optimizer, num_epochs=200)
        all_models.append(trained_model)

        # Make predictions from validation data
        y_valid_pred = model_predict(trained_model,X_valid_k)
        num_errors = len(np.argwhere(y_valid_pred != y_valid_k))
        error_prob[deg-1,k] = num_errors/len(y_valid_k)
        k += 1


  error_prob_m = np.mean(error_prob, axis=1)

  # +1 as the index starts from 0 while the degrees start from 1
  optimal_d = np.argmin(error_prob_m) + 1
  #print("The model selected to best fit the data without overfitting is: d={}".format(optimal_d))
  optimal_hit = error_prob_m[optimal_d-1]
  print("Perceptrons: ", optimal_d, "  Probability of Error: ",optimal_hit)

  return optimal_d, error_prob_m



# %%
class TwoLayerMLP(nn.Module):
    # The nn.CrossEntropyLoss() loss function automatically performs a log_softmax() to 
    # the output when validating, on top of calculating the negative-log-likelihood using 
    # nn.NLLLoss(), while also being more stable numerically... So don't implement from scratch
    # https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    
    def __init__(self, d_in, d_hidden, C):
        super(TwoLayerMLP, self).__init__()
        
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, C)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)  # fc to perceptrons
        x = self.relu(x) # or self.softplus(x) for smooth-ReLU, empirically worse than ReLU
        x = self.fc2(x)  # connect to output layer
        x = self.log_softmax(x)  # for outputs that sum to 1
        return x

# %%
def main():
    # Set Gaussian PDF params, Generate Data, Plot the original data and their true labels
    N = 1000
    n = 3 # dimensionality of input random vectors
    C = 4 # number of classes

    gmm_pdf = {}
    gmm_pdf['priors'] = np.array([1/C, 1/C, 1/C, 1/C])
    mu0 = [2,  2,  2]
    mu1 = [2, -2,  2]
    mu2 = [2, -2, -2]
    mu3 = [2,  2, -2]
    gmm_pdf['m'] = np.array([mu0, mu1, mu2, mu3])
    gmm_pdf['C'] = np.array([2*np.eye(n), 2*np.eye(n), 2*np.eye(n), 2*np.eye(n)])

    X, labels = generate_data_from_gmm(N, gmm_pdf)
    fig = plt.figure()
    ax_gmm = fig.add_subplot(111, projection='3d')

    ax_gmm.plot(X[labels == 0, 0], X[labels == 0, 1], X[labels == 0, 2],'r.', label="Class 1", markerfacecolor='none')
    ax_gmm.plot(X[labels == 1, 0], X[labels == 1, 1], X[labels == 1, 2],'bo', label="Class 2", markerfacecolor='none')
    ax_gmm.plot(X[labels == 2, 0], X[labels == 2, 1], X[labels == 2, 2],'g^', label="Class 3", markerfacecolor='none')
    ax_gmm.plot(X[labels == 3, 0], X[labels == 3, 1], X[labels == 3, 2],'ys', label="Class 4", markerfacecolor='none')
    ax_gmm.set_xlabel(r"$x_1$")
    ax_gmm.set_ylabel(r"$x_2$")
    ax_gmm.set_ylabel(r"$x_3$")

    plt.title("Data and True Class Labels")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # %%
    # Calculate Theoretical Optimal P(Error)
    N = 100000
    X, labels = generate_data_from_gmm(N, gmm_pdf)

    # If 0-1 loss then yield MAP decision rule, else ERM classifier
    Lambda = np.ones((C, C)) - np.eye(C)

    # ERM decision rule, take index/label associated with minimum conditional risk as decision (N, 1)
    decisions = perform_erm_classification(X, Lambda, gmm_pdf, C)

    num_errors = len(np.argwhere(decisions != labels))
    error_prob_th = num_errors/len(labels)
    print("Theoretically Optimal Misclassification Probability = ",error_prob_th)

    # %%
    X_test, y_test = generate_data_from_gmm(100000,gmm_pdf)

    # %%
    fig = plt.figure()

    degs = np.arange(1, 50, 1) # Reduce this to speed up

    N = [100,200,500,1000,2000,5000]
    err_prob_est = []

    for samples in N:
        X_train, y_train = generate_data_from_gmm(samples, gmm_pdf)
        op_d, err_prob_m = model_order_selection(X_train, y_train, folds=10, poly_deg=len(degs)+1)
        plt.plot(degs,err_prob_m,label=samples)

        model = TwoLayerMLP(X_train.shape[1], op_d, C)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        trained_model = train_model(model, X_train, y_train, criterion, optimizer, num_epochs=200)
        y_test_pred = model_predict(trained_model,X_test)


        num_errors = len(np.argwhere(y_test_pred != y_test))
        err_prob_est.append(num_errors/len(y_test))
        print("Misclassification Probability = ",err_prob_est)

    plt.title("P(Error) of Num Perceptrons for each Training Set")
    plt.legend()
    plt.xlabel("Num Perceptrons")
    plt.ylabel("Misclassification Probability")
    plt.show()
    # 

    # %%
    fig = plt.figure()
    plt.scatter(N,err_prob_est)
    plt.plot(N,err_prob_est)
    plt.hlines(y=error_prob_th,xmin=0, xmax=samples,colors="purple", linestyles='--', lw=2, label='Theoretical Optimum from True-Label PDF')
    plt.title("P(Error) for each Training Set using optimal Num of Perceptrons", fontsize=16)
    plt.legend()
    plt.xscale('log')
    plt.ylim(0.125,0.2)
    plt.xlabel("Num Samples", fontsize=14)
    plt.ylabel("Misclassification Probability", fontsize=14)
    plt.show()


if __name__ == '__main__':
    main()