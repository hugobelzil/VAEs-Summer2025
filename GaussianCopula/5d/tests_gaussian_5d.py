import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.copula.api import GaussianCopula

# GETTING THE TRAINING DATA
data = np.load('training_data_gaussian_5d.npy')
samples  = np.load('samples_vae_gaussian_5d.npy')

plt.scatter(data[:,0], data[:,1], s=10, marker='x')
plt.title('Training data used for the VAE (Gaussian copula)')
plt.xlabel(r'$X_1$')
plt.ylabel(r'$X_2$')
plt.savefig('training_data_gaussian_5d.png')
plt.show()

#PLOTTING DATA SAMPLED FROM THE VAE
plt.scatter(samples[:,0], samples[:,1], s=10, marker='x')
plt.title('Sampled data from the trained VAE')
plt.xlabel(r'$X_1$')
plt.ylabel(r'$X_2$')
plt.savefig('samples_vae_gaussian_5d.png')
plt.show()

# TEST 1 : FITTING A COPULA TO THE VALID DATA
mask = (samples >= 0) & (samples <= 1)
mask = mask.all(axis=1)
valid_samples = samples[mask]
plt.scatter(valid_samples[:,0], valid_samples[:,1], s=10, marker='x')
plt.title(r'Sampled data that lies in $[0,1]$')
plt.show()

copula = GaussianCopula(k_dim=5)
estimated_correlation = copula.fit_corr_param(valid_samples)
true_corr = np.array([
    [ 1.00, -0.19,  0.44, -0.01, -0.06],
    [-0.19,  1.00, -0.53, -0.29,  0.76],
    [ 0.44, -0.53,  1.00,  0.56, -0.34],
    [-0.01, -0.29,  0.56,  1.00, -0.34],
    [-0.06,  0.76, -0.34, -0.34,  1.00]
])
print('Estimated correlation : \n',np.round(estimated_correlation,4))
print('True correlation parameter : \n', true_corr)

# TEST 2 : ASSESSING UNIFORMITY OF THE MARGINS
plt.hist(samples[:,0], bins='auto', edgecolor='k', facecolor='g', alpha=0.7)
plt.title('Marginal density of the first component of sampled data')
plt.show()

plt.hist(samples[:,1], bins='auto', edgecolor='k', facecolor='g', alpha=0.7)
plt.title('Marginal density of the second component of sampled data')
plt.show()

