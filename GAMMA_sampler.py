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
def get_input_sfe(m_DM_final):
    return sfe / (m_DM_final / m_DM_0_ref)**sfe_m_index
def get_input_mass_loading(m_DM_final):
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

sampled_points = 10000 #increase as going on with testing
dimensions = 10 #aka number of parameters

#variable editing and range specification
#first column = minimum of variable range
#second column = maximum of variable range - minimum of variable range
var_range = np.zeros([dimensions,2])

#minimum of certain variables, if not zero
var_range[1][0] = 1.0 #f_t_ff
var_range[2][0] = -1.0 #sfe_m_index
var_range[3][0] = .001 #sfe
var_range[4][0] = .05 #f_dyn
var_range[9][0] = .0008 #nb_1a_per_m

#normalization factor of all variables = max - min
var_range[0][1] = 2.0 #t_ff_index
var_range[1][1] = 10.0 - var_range[1][0] #f_t_ff
var_range[2][1] = 1.0 - var_range[2][0] #sfe_m_index
var_range[3][1] = .1 - var_range[3][0] #sfe
var_range[4][1] = .5 - var_range[4][0] #f_dyn
var_range[5][1] = 2.0 #t_sf_z_dep
var_range[6][1] = 2.0 #exp_ml
var_range[7][1] = 2.0 #mass_loading
var_range[8][1] = 1.0 #f_halo_to_gal_out
var_range[9][1] = .002 - var_range[9][0] #nb_1a_per_m

comm.Barrier()

# set size of local data size as well as the big output array
num_gal = len(sub_trees)+1

if comm.rank == 0:
    lhd_temp = lhs(dimensions, samples=sampled_points)
    lhd = lhd_temp #redundant cell in case above is accidentally run again. come up with a better solution (permanent random seed)
    emsp_gather = np.empty([sampled_points, dimensions])
    gal_Mstar = np.empty([sampled_points, num_gal])
    gal_FeH_mean = np.empty([sampled_points, num_gal])
    gal_FeH_std = np.empty([sampled_points, num_gal])
    em_sample_points = np.copy(lhd)
    for i in range(0, sampled_points):
        for dim in range(0,dimensions):
            em_sample_points[i][dim] = (lhd[i][dim]*(var_range[dim][1]))+var_range[dim][0]
    np.save("em_sample_points"+str(sampled_points)+"_3.npy", em_sample_points)
else:
    em_sample_points = None
    emsp_gather = None
    gal_Mstar = None
    gal_FeH_mean = None
    gal_FeH_std = None
    em_sample_points = None

#created on all processors
loc_samples = int(sampled_points/comm.size)
mini_emsp = np.empty([loc_samples, dimensions])
mini_gal_Mstar = np.empty([loc_samples, num_gal])
mini_gal_FeH_mean = np.empty([loc_samples, num_gal])
mini_gal_FeH_std = np.empty([loc_samples, num_gal])

comm.Barrier()

comm.Scatter(em_sample_points, mini_emsp)
'''
for r in range(comm.size):
    if comm.rank == r:
        print("[%d] %s" % (comm.rank, mini_emsp)
    comm.Barrier()
'''
#filename = "debug_statements_#"+str(comm.rank)

#debug = open(filename,"a")

#each i represents a run of GAMMA
sample_start = time.time()
for i in range(0, loc_samples):  
    print("GAMMA sample #: "+str(comm.rank)+"."+str(i))
    start = time.time()

    #setting the GAMMA parameters for a run of GAMMA
    t_ff_index = mini_emsp[i][0]
    f_t_ff = mini_emsp[i][1]
    sfe_m_index = mini_emsp[i][2]
    sfe = mini_emsp[i][3]
    f_dyn = mini_emsp[i][4]
    t_sf_z_dep = mini_emsp[i][5]
    exp_ml = mini_emsp[i][6]
    mass_loading = mini_emsp[i][7]
    f_halo_to_gal_out = mini_emsp[i][8]
    nb_1a_per_m = mini_emsp[i][9]
    
    #debug.write("Parameters: "+ str(t_ff_index)+", "+str(f_t_ff)+", "+str(sfe_m_index)+", "+str(sfe)+", "+str(f_dyn)+", "+str(t_sf_z_dep)+", "+str(exp_ml)+", "+str(mass_loading)+", "+str(f_halo_to_gal_out)+", "+str(nb_1a_per_m))
    #debug.write("\n")
    # Define the dictionary containing GAMMA parameters
    #gamma_kwargs = {"print_off":True, "t_ff_index":t_ff_index, "f_t_ff":f_t_ff,\
     #           "sfe_m_index":sfe_m_index, "sfe":sfe, "f_dyn":f_dyn,\
      #          "t_sf_z_dep":t_sf_z_dep, "exp_ml":exp_ml, "mass_loading":mass_loading,\
       #         "f_halo_to_gal_out":f_halo_to_gal_out, "nb_1a_per_m":nb_1a_per_m,\
        #        "C17_eta_z_dep":C17_eta_z_dep, "DM_outflow_C17":DM_outflow_C17,\
         #       "sfe_m_dep":sfe_m_dep, "t_star":t_star, "t_inflow":t_inflow}
    
    gamma_kwargs = {"print_off":True, "t_ff_index":t_ff_index, "f_t_ff":f_t_ff,\
                "sfe_m_index":sfe_m_index, "sfe":get_input_sfe(host_tree.m_DM_0), \
                "f_dyn":f_dyn, "t_sf_z_dep":t_sf_z_dep, "exp_ml":exp_ml, \
                "mass_loading":get_input_mass_loading(host_tree.m_DM_0),\
                "f_halo_to_gal_out":f_halo_to_gal_out, "nb_1a_per_m":nb_1a_per_m,\
                "C17_eta_z_dep":C17_eta_z_dep, "DM_outflow_C17":DM_outflow_C17,\
                "sfe_m_dep":sfe_m_dep, "t_star":t_star, "t_inflow":t_inflow}
    #debug.write(str(gamma_kwargs)+"\n")
    #debug.write("\n")
    # Run GAMMA for the host tree
    ghost = run_gamma(host_tree, mvir_thresh, gamma_kwargs, SSPs_in)
    comm.Barrier()
    # Run GAMMA for the every sub-trees 
    gsubs = []
    for i_sub in range(len(sub_trees)):
        gamma_kwargs["sfe"] = get_input_sfe(sub_trees[i_sub].m_DM_0)
        gamma_kwargs["mass_loading"] = get_input_mass_loading(sub_trees[i_sub].m_DM_0)
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
    
    #x = np.isnan(mini_gal_Mstar)
    
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
