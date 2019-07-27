import numpy as np
import os
import sys
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as mse
from joblib import dump, load
import emcee #version 2.2.x
from emcee.utils import MPIPool
import corner
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import math
import copy

cwd = os.getcwd()

#emulator models (stand in for actual GAMMA model)
#4 sec to load in
stellar_mass_em = load(cwd+"/Emulator_results/stellar_mass_emulator.joblib")
metallicity_em = load(cwd+"/Emulator_results/metallicity_emulator.joblib")

#IMPORTANT NOTE: should not try to do print statements with the following syntax, otherwise code will seem to be running when it's not
#with MPIPool() as pool:
#    if pool.is_master():
#        print("MCMC run")
print("Emulators loaded in")

#load in training emulator sample points for count of parameters
emsp_dict = np.load(cwd+"/samples_GAMMA/em_sample_points200.npy")
emsp = np.zeros([len(emsp_dict), len(emsp_dict[0])])
i = 0
for i in range(len(emsp_dict)):
    j = 0
    for key in emsp_dict[0].keys():
        emsp[i,j] = copy.deepcopy(emsp_dict[i][key])
        j += 1
dimcount = len(emsp[0])

#load in observational data
obs = np.loadtxt(open(cwd+"/Observations/Metallicity_Mstar_values.csv", "rb"), delimiter=",", skiprows=1, usecols = (0,2,3,4))
obs_Mstar = np.array(np.log10(obs[:,1]))
obs_FeH = obs[:,2]
obs_FeH_error = obs[:,3]

#find line of best fit through the observation points
obs_fit = LR().fit(obs_Mstar.reshape(-1,1),obs_FeH)
obs_pred_FeH = obs_fit.predict(obs_Mstar.reshape(-1,1))
#error from the McConnachie paper, then the mean squared error around the line of best fit
sys_var_FeH = np.mean(obs_FeH_error)**2
reg_var_FeH = mse(obs_FeH, obs_pred_FeH)

print("Observations loaded in")

#specify variable ranges
var_range = np.zeros([dimcount,2]) # should eventually pull the ranges from GAMMA sampler script
var_range[0] = [0.0,2.0]
var_range[1] = [1.0,10.0]
var_range[2] = [-1.0,1.0]
var_range[3] = [.001,.1]
var_range[4] = [.05,.5]
var_range[5] = [0.0,2.0]
var_range[6] = [0.0,2.0]
var_range[7] = [0.0,2.0]
var_range[8] = [0.0,1.0]
var_range[9] = [.0008,.002]


#log-likelihood function
#calculate log_prob between APOGEE observation and model results
#DOES NOT currently include the host galaxy in the fit
def log_red_chi_2(position):
    for i in range(len(var_range)):
        if not var_range[i][0] < position[i] < var_range[i][1]:
            return -np.inf
    calc_Mstar = np.zeros([len(stellar_mass_em)-1]) #Change len(stellar_mass_em)-1 to len(stellar_mass_em) to include host galaxy
    calc_FeH = np.zeros([len(stellar_mass_em)-1]) #see above
    obs_FeH = np.zeros([len(stellar_mass_em)-1]) #see above
    pos = np.array(position) #sklearn won't take in the position array unless it's been wrapped in another array
    for i in range(len(stellar_mass_em)-1):
        calc_Mstar[i] = stellar_mass_em[i].predict(pos.reshape(1,-1))
        calc_FeH[i] = metallicity_em[i].predict(pos.reshape(1,-1))
        obs_FeH[i] = obs_fit.predict(calc_Mstar[i].reshape(1,-1))
    chi_2_FeH = -0.5*(np.sum(((obs_FeH-calc_FeH)**2/(reg_var_FeH+sys_var_FeH))))
    return float(chi_2_FeH)

print("Begin MCMC run")

#begins MCMC run in parallel
with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
   #these starting values are just about the average of the current proposed range of values
    start = np.zeros((dimcount,))
    for i in range(len(start)):
        start[i] = (var_range[i][0]+var_range[i][1])/2
    #number of parameters and walkers,
    ndim, nwalkers, steps  = dimcount, 200, 50000
    #spreads walkers out into small ball
    pos = [start + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    if pool.is_master():
        print("Walkers: "+str(nwalkers)+", Steps per Walker: "+str(steps))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_red_chi_2, pool=pool)
    sampler.run_mcmc(pos, steps)
print("Finished MCMC run")


#saving samples, likelihoods, and the walker's acceptance fractions
samples = sampler.chain[:, :, :].reshape((-1, ndim))
name = str(nwalkers)+"w"+str(steps)+"s"
dump(samples, cwd+"/MCMC_results/samples_"+str(name)+".joblib")
likelihood = sampler.lnprobability
dump(likelihood, cwd+"/MCMC_results/likelihood_"+str(name)+".joblib")
af = open(cwd+"/MCMC_results/acceptance_frac_"+str(name)+".txt", "w+")
af.write(str(sampler.acceptance_fraction))
af.close()
