#BC
# Import python modules
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy
import imp
import os
import glob
import sys
import time
from pyDOE import *
from mpi4py import MPI

#BC
# Define path to the codes
sygmadir = '/mnt/home/f0008572/Chem_Evol_Code/NuPyCEE'
jinapydir = '/mnt/home/f0008572/Chem_Evol_Code/JINAPyCEE'
cagadir = '/mnt/home/f0008572/Chem_Evol_Code/caga/caga'

# Import the codes
caga  = imp.load_source('caga', cagadir+'/caga.py')
calc  = imp.load_source('calc', cagadir+'/calc.py')
plot  = imp.load_source('plot', cagadir+'/plot.py')
sygma = imp.load_source('sygma', sygmadir+'/sygma.py')
omega = imp.load_source('omega', sygmadir+'/omega.py')
gamma = imp.load_source('gamma', jinapydir+'/gamma.py')

comm = MPI.COMM_WORLD

if comm.rank == 0:
    print("-"*78)
    print(" Running on %d cores" % comm.size)
    print("-"*78)

comm.Barrier()
#BC
# Get the file name for the host tree (Milky-Way-like halo)
hostID = 686
hostfname = cagadir+"/../notebooks/H1725272_LX11/rsid{}.npy".format(hostID)

# Get the file names for the satellite sub-trees
# Below, the host tree will be removed from that list
subfnames = glob.glob(cagadir+"/../notebooks/H1725272_LX11/*")

# Convert \ into /, (happens sometime with Windows machines) - not need on HPCC
'''
for i in range(len(subfnames)):
    i_char = subfnames[i].index("\\")
    temp = ""
    for j in range(i_char):
    i_char = subfnames[i].index("\\")
    temp = subfnames[i][:i_char]
    temp += "/"
    temp += subfnames[i][i_char+1:]
    subfnames[i] = temp
    print(subfnames[i])
'''
subfnames.remove(hostfname)
print(len(subfnames),'sub-trees found')

#BC
# Load the GAMMA input arrays for each tree
host_tree = caga.gamma_tree.load(hostfname)
sub_trees = [caga.gamma_tree.load(subfname) for subfname in subfnames]

#BC
# Precalculate stellar populations to accelerate GAMMA computing time
SSPs_in = caga.precompute_ssps()

comm.bcast(SSPs_in)
comm.Barrier()

#BC
# Function to gather paramaters and run GAMMA
def run_gamma(gt, mvir_thresh, gamma_kwargs, SSPs_in):
    kwargs = caga.generate_kwargs(gt, mvir_thresh, SSPs_in=SSPs_in)
    kwargs.update(gamma_kwargs)
    g = gamma.gamma(**kwargs)
    return g

# Function to return the correctly-scaled parameters
# for trees, as a function of their final dark matter mass
def get_input_sfe(sfe, m_DM_final, sfe_m_index):
    print(sfe)
    return sfe / (m_DM_final / m_DM_0_ref)**sfe_m_index
def get_input_mass_loading(mass_loading, m_DM_final, exp_ml):
    print(mass_loading)
    c_eta = mass_loading * m_DM_0_ref**(exp_ml/3.0)
    return c_eta * m_DM_final**(-exp_ml/3.0)

#BC
# Parameters that SHOULD NOT be modified
# They are to activate certain parts of the code
# to accept the parameters mentionned above
C17_eta_z_dep = False
DM_outflow_C17 = True
sfe_m_dep = True
t_star = -1
t_inflow = -1

# Set the dark matter halo at redshift 0 where
# the parameters will be refering to.
# SHOULD NOT be modified
m_DM_0_ref = 1.0e12


# Minimium virial halo mass below which no star formation can occur
# This will likely disapear soon, so should not be part of the Gaussian process
mvir_thresh = 3e7

#variable editing and range specification - dictionary
# Parameters to be sampled
# var_range['dictionary_key'][0] -- minimum parameter value
# var_range['dictionary_key'][1] -- range (max - min)
var_range = {}
var_range["t_ff_index"] = [0.0, 2.0]
var_range["f_t_ff"] = [1.0, 9.0]
var_range["sfe_m_index"] = [-1.0, 2.0]
var_range["sfe"] = [.001, .099]
var_range["f_dyn"] = [.05, .45]
var_range["t_sf_z_dep"] = [0.0,2.0]
var_range["exp_ml"] = [0.0,2.0]
var_range["mass_loading"] = [0.0,2.0] 
var_range["f_halo_to_gal_out"] = [0.0,1.0]
var_range["nb_1a_per_m"] = [0.8e-3,1.2e-3]

sampled_points = 10 #increase as going on with testing
dimensions = len(var_range) #aka number of parameters

comm.Barrier()

# set size of local data size as well as the big output array
num_gal = len(sub_trees)+1

if comm.rank == 0:
    lhd_temp= lhs(dimensions, samples=sampled_points)
    lhd = []
    emsp_gather = np.empty([sampled_points, dimensions])
    gal_Mstar = np.empty([sampled_points, num_gal])
    gal_FeH_mean = np.empty([sampled_points, num_gal])
    gal_FeH_std = np.empty([sampled_points, num_gal])
    emsp_index = np.arange(sampled_points)
    em_sample_points = []
    for i in range(0, sampled_points):
        j = 0
        lhd.append({})
        em_sample_points.append({})
        for key in var_range.keys():
            lhd[i][key] = lhd_temp[i][j]
            j += 1
    np.save("em_sample_points"+str(sampled_points)+"_3.npy", em_sample_points)
else:
    lhd_temp = None
    lhd = None
    emsp_gather = None
    gal_Mstar = None
    gal_FeH_mean = None
    gal_FeH_std = None
    em_sample_points = None
    emsp_index = None

#created on all processors
loc_samples = int(sampled_points/comm.size)
mini_emsp_index = np.empty([loc_samples])
mini_gal_Mstar = np.empty([loc_samples, num_gal])
mini_gal_FeH_mean = np.empty([loc_samples, num_gal])
mini_gal_FeH_std = np.empty([loc_samples, num_gal])

comm.Barrier()
comm.bcast(em_sample_points)
comm.Barrier()
comm.Scatter(emsp_index, mini_emsp_index)
comm.Barrier()

#filename = "debug_statements_#"+str(comm.rank)

#debug = open(filename,"a")

#each i represents a run of GAMMA
sample_start = time.time()
for i in np.nditer(mini_emsp_index):  
    print("GAMMA sample #: "+str(i))
    start = time.time()

    #setting the GAMMA parameters for a run of GAMMA
    # Get the default "empty" list of parameters
    gamma_kwargs = {"print_off":True, "C17_eta_z_dep":C17_eta_z_dep, \
                      "DM_outflow_C17":DM_outflow_C17, "t_star":t_star, \
                      "t_inflow":t_inflow, "sfe_m_dep":sfe_m_dep }

    # Add the sampled parameters
    for key in var_range.keys():
        gamma_kwargs[key] = copy.deepcopy(em_sample_points[i][key])
    
    gamma_kwargs["sfe"] = get_input_sfe(em_sample_points[i]["sfe"],host_tree.m_DM_0, em_sample_points[i]["sfe_m_index"])
    gamma_kwargs["mass_loading"] = get_input_mass_loading(em_sample_points[i]["mass_loading"], host_tree.m_DM_0, em_sample_points[i]["exp_ml"])

    #debug.write(str(gamma_kwargs)+"\n")
    #debug.write("\n")
    # Run GAMMA for the host tree
    ghost = run_gamma(host_tree, mvir_thresh, gamma_kwargs, SSPs_in)
    comm.Barrier()
    # Run GAMMA for the every sub-trees 
    gsubs = []
    for i_sub in range(len(sub_trees)):
        gamma_kwargs_1["sfe"] = get_input_sfe(em_sample_points[i]["sfe"], sub_trees[i_sub].m_DM_0, em_sample_points[i]["sfe_m_index"])
        gamma_kwargs_1["mass_loading"] = get_input_mass_loading(em_sample_points[i]["mass_loading"], sub_trees[i_sub].m_DM_0,  em_sample_points[i]["exp_ml"])
        gsubs.append(run_gamma(sub_trees[i_sub], mvir_thresh, gamma_kwargs, SSPs_in))

    comm.Barrier()
    #========================
    ## Extraction of outputs
    #========================
    # add output processing here and put it in an array where host galaxy is very last entry in each row, ie. x[i][-1] should be the host galaxy value
    
    len_gsubs = len(gsubs)
    
    # Extract the final (uncorrected) stellar mass of each tree
    for j in range(len_gsubs):
        mini_gal_Mstar[i][j] = calc.mstar_evolution(gsubs[j])[-1]#check on this output
    mini_gal_Mstar[i][-1] = calc.mstar_evolution(ghost)[-1]

    comm.Barrier()
    # Extract the metallicity distribution function (MDF) of each tree
    # There might be warnings, but it is ok
    host_mdf = calc.mdf(ghost)
    sub_mdfs = [calc.mdf(g) for g in gsubs]
    comm.Barrier()

    # Extract the average and standard deviation of metallicity [Fe/H] of each tree
    subs_FeH_mean = [caga.find_distribution_mean(*sub_mdf) for sub_mdf in sub_mdfs]
    subs_FeH_std = [caga.find_distribution_std(*sub_mdf) for sub_mdf in sub_mdfs]
    comm.Barrier()

    for j in range(len_gsubs):
        mini_gal_FeH_mean[i][j] = subs_FeH_mean[j]
        mini_gal_FeH_std[i][j] = subs_FeH_std[j]
    mini_gal_FeH_mean[i][-1] = caga.find_distribution_mean(*host_mdf)
    mini_gal_FeH_std[i][-1] = caga.find_distribution_std(*host_mdf)
    
    print("Total GAMMA sample #: "+str(comm.rank)+"."+str(i)+"run time is {:.1f}".format(time.time()-start))
    comm.Barrier()
    
print("Total sampling #:"+str(comm.rank)+"."+str(i)+"time is {:.1f}".format(time.time()-sample_start))

comm.Barrier()
comm.Gather(mini_gal_Mstar, gal_Mstar)
comm.Gather(mini_gal_FeH_mean, gal_FeH_mean)
comm.Gather(mini_gal_FeH_std, gal_FeH_std)

if comm.rank == 0:
    np.save("gal_Mstar_"+str(sampled_points)+"_3.npy", gal_Mstar)
    np.save("gal_FeH_mean_"+str(sampled_points)+"_3.npy", gal_FeH_mean)
    np.save("gal_FeH_std_"+str(sampled_points)+"_3.npy", gal_FeH_std)

#debug.close()

'''
j = 0
fig, ax = plt.subplots(2, 5, sharex='col', sharey='row')
for i in range(1):
    for k in range(2):
        ax[i, k].errorbar(gal_Mstar[j][:-1], gal_FeH_mean[j][:-1], yerr=gal_FeH_std[j][:-1], fmt='o', label='sub-trees')
        ax[i, k].errorbar(gal_Mstar[j][-1], gal_FeH_mean[j][-1], yerr=gal_FeH_std[j][-1], fmt='o', label='host-tree')
        ax[i, k].set_xscale('log')
        j += 1

plt.legend(loc='right')
#plt.xscale('log')
fig.text(0.5, 0.03, '$M_\star$ [M$_\odot$]', ha='center', va='center')
fig.text(0.06, 0.5, '[Fe/H]', ha='center', va='center', rotation='vertical')
fig.subplots_adjust(hspace=0,wspace = 0)
fig.suptitle('Metallicity [Fe/H] vs Stellar Mass [M$_\odot$]')
plt.legend(bbox_to_anchor=(-2,4.04,1.5,1), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2)

plt.show()
'''
