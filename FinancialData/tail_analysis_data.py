import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import powerlaw
#checking if the data is heavy tailed
data = np.load("data/train_financial_dataset.npy")
eval_data = np.load("data/eval_financial_dataset.npy")

# LOOP TO GET PARAMETERS OF DISTRIBUTION OF COMPONENTS:
for i in range(data.shape[1]):
    print(f"COMPONENT {i+1}")
    mu, sigma = stats.norm.fit(data[:,i])
    print(f"mu: {np.round(mu,4)}, sigma: {np.round(sigma,4)}")
    x = np.linspace(np.min(data[:,i]),np.max(data[:,i]),300)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.hist(eval_data[:,i],bins=40,edgecolor = 'black', density=True)
    plt.title(f"Fitted normal density to the {i+1}-th marginal")
    plt.show()
    kurtosis_sample = stats.kurtosis(eval_data[:,i],fisher=True)
    print('Excess Kurtosis: ', kurtosis_sample)

    df, loc, scale = stats.t.fit(eval_data[:,i])
    plt.plot(x, stats.t.pdf(x, df, loc, scale))
    plt.hist(eval_data[:, i], bins=40, edgecolor='blue', density=True)
    plt.title(f"Fitted Student's t density to the {i + 1}-th marginal")
    plt.show()
    print('df = ',df)
    sm.qqplot(data[:,i],dist=stats.t, distargs=(df,), loc=loc, scale=scale, line='45')
    plt.title("QQ-plot vs Fitted Student's t-distribution")
    plt.grid(True)
    plt.show()

    fit = powerlaw.Fit(eval_data[:,i])
    print("index : ",fit.alpha)
