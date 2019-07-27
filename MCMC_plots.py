import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.linear_model import LinearRegression as LR
from joblib import load
import emcee
import corner
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
print("Imported packages")

#specify walker number, step number, and import MCMC samples and likelihoods
cwd = os.getcwd()
ndim, nwalkers = 10, 200
steps = 50000
burnin = 5000
name = str(nwalkers)+"w"+str(steps)+"s"
samples = load(cwd+"/MCMC_results/samples_"+str(name)+".joblib")
likelihood = load(cwd+"/MCMC_results/likelihood_"+str(name)+".joblib")

print("Imported samples and likelihoods")

#plt.clf()

#plot variable traces
labels=["GIR Redshift Dependency", "GIR Coefficient", "SFE Dark Matter Dependency", "SFE Coefficient", "SFT Coefficient", "SFT Redshift Depedency", "GOR Dark Matter Dependency", "GOR Coefficient", "COR Coefficient", "# of 1a SNE per Mstar"]

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 9))
for i in range(len(axes)):
    axes[i].plot(samples[:, i].T, color="k", alpha=0.4)
    axes[i].yaxis.set_major_locator(MaxNLocator(5))
    axes[i].set_xlabel(labels[i])

print("Created Variable traces")
fig.tight_layout(h_pad=0.0)
fig.savefig(cwd+"/MCMC_results/trace_"+str(name)+".png")
plt.clf()

#plot likelihood chain of first walker
index = np.arange(0,len(likelihood[0]))
plt.plot(index, likelihood[0])
plt.savefig(cwd+"/MCMC_results/likelihood_"+str(name)+".png")

print("Likelihood plots created")

plt.clf()

#create corner plot + pdfs
median_sample = np.median(samples, axis=0)
hist2d_kwargs = {"rasterized": True}
fig = corner.corner(samples[burnin:], labels=labels, truths = median_sample.tolist(), truth_color='#C28E0E', **hist2d_kwargs)
fig.savefig(cwd+"/MCMC_results/triangle_pdfs_"+str(name)+".png", dpi=125)
print("Corner Plot created")

#save the median of each PDF to a file
value_results = open(cwd+"/MCMC_results/best_fit_values_"+str(name)+".txt", "w+")
value_results.write(str(median_sample))
value_results.close()
