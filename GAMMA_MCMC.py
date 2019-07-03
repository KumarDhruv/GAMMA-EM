import numpy as np
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import sys
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.linear_model import LinearRegression as LR
from joblib import dump, load
import emcee
from emcee.utils import MPIPool
import corner
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

cwd = os.getcwd()

#emulator models (stand in for actual GAMMA model)
#4 sec to load in
stellar_mass_em = load(cwd+"/stellar_mass_emulator.joblib")
metallicity_em = load(cwd+"/metallicity_emulator.joblib")

print("Emulators loaded in")

obs = np.loadtxt(open(cwd+"/Observations/McConnachie Paper/McConnahie Metallicity and Mstar values.csv", "rb"), delimiter=",", skiprows=1, usecols = (0,2,3,4))
obs_Mstar = np.array(obs[:,1])
obs_FeH = obs[:,2]
obs_FeH_error = obs[:,3]
var_FeH = np.mean(obs_FeH_error)**2

obs_fit = LR().fit(obs_Mstar.reshape(-1,1),obs_FeH)
var_range = np.zeros([10,2]) # should eventually pull this from GAMMA sampler
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

print("Observations loaded in")

#log-likelihood function
#calculate log_prob between APOGEE observation and model results

def log_red_chi_2(position,ndim):
    for i in range(len(var_range)):
        if not var_range[i][0] < position[i] < var_range[i][1]:
            return -np.inf
    calc_Mstar = np.zeros([len(stellar_mass_em)])
    calc_FeH = np.zeros([len(stellar_mass_em)])
    obs_FeH = np.zeros([len(stellar_mass_em)])
    pos = np.array(position)
    print(pos)
    for i in range(len(stellar_mass_em)):
        calc_Mstar[i] = stellar_mass_em[i].predict(pos.reshape(1,-1))
        calc_FeH[i] = metallicity_em[i].predict(pos.reshape(1,-1))
        obs_FeH[i] = obs_fit.predict(calc_Mstar[i].reshape(1,-1))
    chi_2_FeH = np.sum(((obs_FeH-calc_FeH)**2/(np.var(obs_FeH-calc_FeH)+var_FeH)))
    red_chi_2_FeH = chi_2_FeH/ndim
    return float(red_chi_2_FeH)

print("Begin MCMC run")

with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    #parameters to fit - eventually should pull this from the GAMMA-EM sampling script
    #these starting values are just about the average of the current proposed range of values
    #t_ff_index, f_t_ff, sfe_m_index, sfe, f_dyn, t_sf_z_dep, exp_ml, mass_loading, f_halo_to_gal_out, nb_1a_per_m
    start = np.array([1.0,5.5,0.1,0.055,0.3,1.0,1.0,1.0,0.5, 1.0e-3])
    
    #number of parameters and walkers, and their starting point
    ndim, nwalkers = len(start), 150
    steps = 8000
    pos = [start + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_red_chi_2, args=[ndim],pool=pool)
    sampler.run_mcmc(pos, steps)
    
print("Finished MCMC run")
flat_samples = sampler.chain[:, 1000:, :].reshape((-1, ndim))
chain_name = str(nwalkers)+"w"+str(steps)+"s.joblib"
dump(flat_samples, chain_name)    
#figure analysis
likelihood = sampler.lnprobability
plt.plot(likelihood).savefig("MCMC-lnlikelihood.png")

plt.clf()

labels=["t_ff_index", "f_t_ff", "sfe_m_index", "sfe", "f_dyn", "t_sf_z_dep", "exp_ml", "mass_loading", "f_halo_to_gal_out", "nb_1a_per_m"]

fig, axes = plt.subplots(10, 1, sharex=True, figsize=(8, 9))
for i in range(len(axes)):
    axes[i].plot(sampler.chain[:, 1000:, i].T, color="k", alpha=0.4)
    axes[i].yaxis.set_major_locator(MaxNLocator(5))
    axes[i].set_ylabel(labels[i])

fig.tight_layout(h_pad=0.0)
fig.savefig("MCMC-time.png")

fig = corner.corner(flat_samples)
print("Figure created")    

fig.savefig("MCMC-VarValues.png")
