import matplotlib.pyplot as plt #General Plotting
from matplotlib import cm
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix

# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(4)

N = 10000
mu = np.array([[-1/2, -1/2, -1/2], [1, 1, 1]])

sigma = np.array([[[1, 0.3, -0.2],
                   [0.3, 1, 0.3],
                   [-0.2, 0.3, 1]],
                  [[1, 0.3, -0.2],
                   [0.3, 1, 0.3],
                   [-0.2, 0.3, 1]]])
                  
priors = np.array([0.65, 0.35])


n = mu.shape[1]
C = len(priors)

print(C)
print(n)

# Decide randomly which samples will come from each component
# np.random.rand creates an array of the given shape and populates it with random
# samples from a uniform distribution over [0,1). Similar to np.zeros or np.ones
u = np.random.rand(N)
print(u.shape)

# set thresholds: 0 -> 0.65 -> 1
thresholds = np.cumsum(priors)
thresholds = np.insert(thresholds,0,0)

# create empty matrix for samples of size 10,000 x 3
X = np.zeros([N,n])

# create empty labels array of length 10,000
labels = np.zeros(N)

# Plot for original data and their true labels
fig = plt.figure(figsize=(12, 10))
ax1 = fig.add_subplot(131, projection = '3d')
marker_shapes = 'd+.'
marker_colors = 'rbg' 

pp1 = []
pp2 = []

for i in range(C):

    # Find indices of u that meet each prior
    indices = np.argwhere((thresholds[i] <= u) & (u <= thresholds[i+1]))[:, 0]
    print(indices)

    # Get the number of indices in each component - should be ~ 6500 and 3500
    Nl = len(indices)
    
    # set label vector based on above - will be vector of class labels [0 0 1 0 1 .. to 9999] in this case
    # for more classes, this can be [1 2 2 0 1 3 1 3 0 ...]
    labels[indices] = i * np.ones(Nl)
    
    # for each valid index, fill the 
    X[indices, :] = multivariate_normal.rvs(mu[i], sigma[i], Nl)
    print(X.shape)
    ax1.plot(X[labels==i, 0], X[labels==i, 1], X[labels==i, 2], marker_shapes[i] + marker_colors[i], label="True Class {}".format(i))


# Plot the original data and their true labels
plt.legend()
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Generated Original Data Samples")
plt.tight_layout()
plt.show()

