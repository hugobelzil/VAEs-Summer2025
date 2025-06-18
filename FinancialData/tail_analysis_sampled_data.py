import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import powerlaw

#checking if the data is heavy tailed
samples = np.load("data/samples_vae.npy")
#eval_data = np.load("data/eval_financial_dataset.npy")

# LOOP TO GET PARAMETERS OF DISTRIBUTION OF COMPONENTS:
for i in range(samples.shape[1]):
    print(f"COMPONENT {i+1}")
    mu, sigma = stats.norm.fit(samples[:,i])
    print(f"mu: {np.round(mu,4)}, sigma: {np.round(sigma,4)}")
    x = np.linspace(np.min(samples[:,i]),np.max(samples[:,i]),300)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.hist(samples[:,i],bins=40,edgecolor = 'black', density=True)
    plt.title(f"Fitted normal density to the {i+1}-th marginal")
    plt.show()
    kurtosis_sample = stats.kurtosis(samples[:,i],fisher=True)
    print('Excess Kurtosis: ', kurtosis_sample)
    sm.qqplot(samples[:, i], dist=stats.norm, loc=mu, scale=sigma, line='45')
    plt.title("QQ-plot vs Fitted Normal distribution")
    plt.grid(True)
    plt.show()

    df, loc, scale = stats.t.fit(samples[:,i])
    plt.plot(x, stats.t.pdf(x, df, loc, scale))
    plt.hist(samples[:, i], bins=40, edgecolor='blue', density=True)
    plt.title(f"Fitted Student's t density to the {i + 1}-th marginal")
    plt.show()
    print('df = ',df)
    sm.qqplot(samples[:,i],dist=stats.t, distargs=(df,), loc=loc, scale=scale, line='45')
    plt.title("QQ-plot vs Fitted Student's t-distribution")
    plt.grid(True)
    plt.show()

    #fit = powerlaw.Fit(samples[:,i])
    #print("index : ",fit.alpha)
