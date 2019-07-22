##Make a different file explaining predicting code
##Make repository with this stuff + sample set
##Save the actual emulator

#importing necessary packages
import numpy as np
from joblib import dump
import time
import os
import copy

#Scikit Gaussian Process functions
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import PairwiseKernel, RBF, Product, ConstantKernel as C, RationalQuadratic as RQ, Matern, WhiteKernel

#Scikit MultiOutput Regression class - not needed in the current version
#from sklearn.multioutput import MultiOutputRegressor

#Scikit Metrics for manual R^2 score
from sklearn.metrics import r2_score

start = time.time()
cwd = os.getcwd()

#load in input training data
emsp_train_dict = np.load(cwd+"/samples_GAMMA/em_sample_points200.npy") #address of set of sample parameter
#load in output training data
gal_Mstar_train = np.load(cwd+"/samples_GAMMA/gal_Mstar_200.npy")
gal_FeH_mean_train = np.load(cwd+"/samples_GAMMA/gal_FeH_mean_200.npy")
gal_FeH_std_train = np.load(cwd+"/samples_GAMMA/gal_FeH_std_200.npy")

#load in input testing data
emsp_test_dict = np.load(cwd+"/samples_GAMMA/em_sample_points10000.npy") #set of sample parameter
#load in output testing data
gal_Mstar_test = np.load(cwd+"/samples_GAMMA/gal_Mstar_10000.npy")
gal_FeH_mean_test = np.load(cwd+"/samples_GAMMA/gal_FeH_mean_10000.npy")
gal_FeH_std_test = np.load(cwd+"/samples_GAMMA/gal_FeH_std_10000.npy")

emsp_train = np.zeros([len(emsp_train_dict), len(emsp_train_dict[0])])
emsp_test = np.zeros([len(emsp_test_dict), len(emsp_test_dict[0])])

for i in range(len(emsp_train_dict)):
    j = 0
    for key in emsp_train_dict[0].keys():
        emsp_train[i,j] = copy.deepcopy(emsp_train_dict[i][key])
        j += 1

i = 0
for i in range(len(emsp_test_dict)):
    j = 0
    for key in emsp_test_dict[0].keys():
        emsp_test[i,j] = copy.deepcopy(emsp_test_dict[i][key])
        j += 1

print("================================")
print("First generation of the emulator")
print("================================")
#Stellar mass
sigma_train_1 = np.zeros(len(gal_Mstar_train[0])) #standard deviation of training set for the kernel
em_Mstar_1 = [] #list of stellar mass emulator objects
log_em_Mstar_pred_1 = np.zeros([len(gal_Mstar_test),len(gal_Mstar_test[0])]) #array of stellar mass predictions of the test data
log_em_Mstar_pred_std_1 = np.zeros([len(gal_Mstar_test),len(gal_Mstar_test[0])])#array of standard deviations of predicted stellar mass at the test points
log_gal_Mstar_train = np.log10(gal_Mstar_train) #log base-10 form of the data to remove the high skewness of the original stellar mass data
#add clarification of shape of data

print("Training stellar mass emulators")
#Stellar mass emulators
for i in range(len(gal_Mstar_train[0])):
    sigma_train_1[i] = np.log10(np.std(gal_Mstar_train[:,i]))
    kern = C(sigma_train_1[i]**2) * RBF() * Matern() + WhiteKernel() #likely needs to be modified when new parameters are availabe
    em_Mstar_1.append(GaussianProcessRegressor(kernel=kern, n_restarts_optimizer=1))
    em_Mstar_1[i].fit(emsp_train,log_gal_Mstar_train[:,i])
    log_em_Mstar_pred_1[:,i], log_em_Mstar_pred_std_1[:,i] = em_Mstar_1[i].predict(emsp_test, return_std = True)
    print(".", end = "")
print()
print("Stellar mass emulator scores :")
for i in range(len(em_Mstar)):
    print(em_Mstar[i].score(emsp_test, gal_Mstar_test, end = ", ")
'''
#Alternatively, could use MultiOutputRegressor, but that limits the information available for model analysis
kern = C(sigma_train_2[i]**2) * RBF() * Matern() + WhiteKernel()
em_Mstar = MultiOutputRegressor(GaussianProcessRegressor(kernel=kern, n_restarts_optimizer=1))
em_Mstar.fit(emsp_train,log_gal_Mstar_train)
log_em_Mstar_pred_1 = em_Mstar.predict(emsp_test) #test predictions

#if you want individual emulator standard deviations from MultiOutputRegressor, use this and ignore the x result
for i in range(len(em_Mstar.estimators_)):
    x, log_em_Mstar_pred_std_1[i]= em_Mstar.estimators_[i].predict(emsp_test, return_std = True)
'''

#Metallicity
sigma_train = np.zeros(len(gal_FeH_mean_train[0]))
em_FeH_mean = []
em_FeH_mean_pred = np.zeros([len(gal_FeH_mean_test),len(gal_FeH_mean_test[0])])
em_FeH_mean_pred_std = np.zeros([len(gal_FeH_mean_test),len(gal_FeH_mean_test[0])])

print("Training metallicity emulators")
#Metallicity emulators                          
for i in range(len(gal_FeH_mean_train[0])):
    sigma_train_1[i] = np.std(gal_FeH_mean_train[:,i])
    kern = C(sigma_train_1[i]**2) * RBF() * PairwiseKernel() #likely needs to be modified when new parameters are availabe
    em_FeH_mean.append(GaussianProcessRegressor(kernel=kern, n_restarts_optimizer=1))
    em_FeH_mean[i].fit(emsp_train,gal_FeH_mean_train[:,i])
    em_FeH_mean_pred[:,i], em_FeH_mean_pred_std[:,i] = em_FeH_mean[i].predict(emsp_test, return_std = True)
    print(".", end = "")
print()
print("Metallicity emulator scores :")
for i in range(len(em_FeH_mean)):
    print(em_FeH_mean[i].score(emsp_test, gal_FeH_mean_test, end = ", ")
'''
#Alternate MultiOutputRegressor method
sigma_train = np.std(gal_FeH_mean_train)
#FeH_mean emulator
kern = C(sigma_train[i]**2) * RBF() * PairwiseKernel()
# to set number of jobs to the number of cores, use n_jobs=-1
em_FeH_mean = MultiOutputRegressor(GaussianProcessRegressor(kernel=kern, n_restarts_optimizer=1))
em_FeH_mean.fit(emsp_train,gal_FeH_mean_train)
#test predictions
em_FeH_mean_pred = em_FeH_mean.predict(emsp_test)
em_FeH_mean_pred_std = np.zeros([22,50])
#alternate way to do individual galaxy predictions
for i in range(len(em_FeH_mean.estimators_)):
    x, em_FeH_mean_pred_std[i] = em_FeH_mean.estimators_[i].predict(emsp_test, return_std = True)
    print("Galaxy #" + str(i) +": "+ str(em_FeH_mean.estimators_[i].score(emsp_test, gal_FeH_mean_test[:,i])))
'''

print("=========================================")
print("Development of 2nd generation of emulator")
print("=========================================")

print("Finding areas of high uncertainty in parameter space")
em_combo_std = log_em_Mstar_pred_1*em_FeH_mean_pred_std #combined standard deviation, equally weighted between outputs

var_index = np.arange(0, 10000, 1).reshape(10000,1) #index count of the sample
indexed_combo_std = np.append(em_combo_std, var_index, axis = 1) #puts an in-array index of standard deviation results matching each sample parameter set
indexed_emsp = np.append(emsp_test, var_index, axis = 1) #puts an in-array index of each sample parameter set

num_pts_add = int(len(emsp_train)*.35)

top_index = np.zeros([len(gal_Mstar_train[0]), num_pts_add]) #creates an array to be filled with the set of indexes that are associated with the num_pts_add highest standard deviations in the test sample set

for i in range(len(indexed_combo_std[0])-1):
    x = indexed_combo_std[indexed_combo_std[:,i].argsort()] #sorts the given column 
    top_index[i] = x[-num_pts_add:,-1].T #gets the largest combined std dev in each column
        
top, counts = np.unique(top_index, return_counts=True) #top is the array of the row index of emsp_test points that need to be added to the emulator

print("Adding training data at points of high uncertainty")
#points to add to the training data
emsp_train_adds = np.zeros([len(top), len(emsp_test[0])]) 
gal_Mstar_train_adds = np.zeros([len(top), len(gal_Mstar_test[0])])
gal_FeH_mean_train_adds = np.zeros([len(top), len(gal_FeH_mean_test[0])])

#selecting the training data from the original test set
for i in range(len(top)):
    index = int(top[i])
    emsp_train_adds[i] = emsp_test[index]
    gal_Mstar_train_adds[i] = gal_Mstar_test[index]
    gal_FeH_mean_train_adds[i] = gal_FeH_mean_test[index]

#add training points
emsp_train_2 = np.append(emsp_train, emsp_train_adds, axis = 0)
gal_Mstar_train_2 = np.append(gal_Mstar_train, gal_Mstar_train_adds, axis = 0)
gal_FeH_mean_train_2 = np.append(gal_FeH_mean_train, gal_FeH_mean_train_adds, axis = 0)

#import new set of test points
emsp_test_2_dict = np.load(cwd+"/samples_GAMMA/em_sample_points10000_2.npy")
gal_Mstar_test_2 = np.load(cwd+"/samples_GAMMA/gal_Mstar_10000_2.npy")
gal_FeH_mean_test_2 = np.load(cwd+"/samples_GAMMA/gal_FeH_mean_10000_2.npy")
gal_FeH_std_test_2 = np.load(cwd+"/samples_GAMMA/gal_FeH_std_10000_2.npy")

emsp_test_2 = np.zeros([len(emsp_test_2_dict), len(emsp_test_2_dict[0])])

i = 0
for i in range(len(emsp_test_2_dict)):
    j = 0
    for key in emsp_test_2_dict[0].keys():
        emsp_test_2[i,j] = copy.deepcopy(emsp_test_2_dict[i][key])
        j += 1

print("=================================")
print("Second generation of the emulator")
print("=================================")
sigma_train_2 = np.zeros(len(gal_Mstar_train_2[0])) 
em_Mstar_2 = []
log_em_Mstar_pred_2 = np.zeros([len(gal_Mstar_test_2),len(gal_Mstar_test_2[0])])
log_em_Mstar_pred_std_2 = np.zeros([len(gal_Mstar_test_2),len(gal_Mstar_test_2[0])])
log_gal_Mstar_train_2 = np.log10(gal_Mstar_train_2)

print("Training stellar mass emulators")
#Stellar mass emulators
for i in range(len(gal_Mstar_train_2[0])):
    sigma_train_2[i] = np.log10(np.std(gal_Mstar_train_2[:,i]))
    #print(sigma_train[i])
    kern = C(sigma_train_2[i]**2) * RBF() * Matern() + WhiteKernel()
    em_Mstar_2.append(GaussianProcessRegressor(kernel=kern, n_restarts_optimizer=1))
    em_Mstar_2[i].fit(emsp_train_2,log_gal_Mstar_train_2[:,i])
    log_em_Mstar_pred_2[:,i], log_em_Mstar_pred_std_2[:,i] = em_Mstar_2[i].predict(emsp_test_2, return_std = True)
    print(".", end="")
print()
print("Stellar mass emulator scores:")
for i in range(len(em_Mstar_2)):
    print(em_Mstar_2[i].score(emsp_test_2, gal_Mstar_test_2, end = ", ")

#refer to first generation section for use of MultiOutputRegressor class
    
sigma_train_2 = np.zeros(len(gal_FeH_mean_train_2[0])) 
em_FeH_mean_2 = []
em_FeH_mean_pred_2 = np.zeros([len(gal_FeH_mean_test_2),len(gal_FeH_mean_test_2[0])])
em_FeH_mean_pred_std_2 = np.zeros([len(gal_FeH_mean_test_2),len(gal_FeH_mean_test_2[0])])

print("Training metallicity emulators")
#Metallicity emulators
for i in range(len(gal_FeH_mean_train_2[0])):
    sigma_train_2[i] = np.std(gal_FeH_mean_train_2[:,i])
    #print(sigma_train[i])
    kern = C(sigma_train_2[i]**2) * RBF() * PairwiseKernel()
    em_FeH_mean_2.append(GaussianProcessRegressor(kernel=kern, n_restarts_optimizer=1))
    em_FeH_mean_2[i].fit(emsp_train_2,gal_FeH_mean_train_2[:,i])
    em_FeH_mean_pred_2[:,i], em_FeH_mean_pred_std_2[:,i] = em_FeH_mean_2[i].predict(emsp_test_2, return_std = True)
    print(".", end="")
print()
print("Metallicity emulator scores:")
for i in range(len(em_FeH_mean_2)):
    print(em_FeH_mean_2[i].score(emsp_test_2, gal_FeH_mean_test_2, end = ", ")

#refer to first generation section for use of MultiOutputRegressor class


##third generation currently not included because of convergence warnings
#print("=========================================")
#print("Development of 3rd generation of emulator")
#print("=========================================")
#
#print("Finding areas of high uncertainty in parameter space")
#em_combo_std = log_em_Mstar_pred_std_2*em_FeH_mean_pred_std_2 #combined standard deviation, equally weighted between outputs
#
#var_index = np.arange(0, 10000, 1).reshape(10000,1) #index count of the sample
#indexed_combo_std = np.append(em_combo_std, var_index, axis = 1) #puts an in-array index of standard deviation results matching each sample parameter set
#indexed_emsp = np.append(emsp_test_2, var_index, axis = 1) #puts an in-array index of each sample parameter set
#
#num_pts_add_2 = int(len(emsp_train_2 * .3))
#
#top_index = np.zeros([len(gal_Mstar_train[0]), num_pts_add_2]) #creates an array to be filled with the set of indexes that are associated with the 75 highest standard deviations in the test sample set
#
#for i in range(len(indexed_combo_std[0])-1):
#    x = indexed_combo_std[indexed_combo_std[:,i].argsort()] #sorts the given column 
#    top_index[i] = x[-num_pts_add:,-1].T #gets the largest combined std dev in each column
#        
#top, counts = np.unique(top_index, return_counts=True) #top is the array of the row index of emsp_test points that need to be added to the emulator
#
#print("Adding training data at points of high uncertainty")
##points to add to the training data
#emsp_train_adds = np.zeros([len(top), len(emsp_test_2[0])]) 
#gal_Mstar_train_adds = np.zeros([len(top), len(gal_Mstar_test_2[0])])
#gal_FeH_mean_train_adds = np.zeros([len(top), len(gal_FeH_mean_test_2[0])])
#
##selecting the training data from the original test set
#for i in range(len(top)):
#    index = int(top[i])
#    emsp_train_adds[i] = emsp_test_2[index]
#    gal_Mstar_train_adds[i] = gal_Mstar_test_2[index]
#    gal_FeH_mean_train_adds[i] = gal_FeH_mean_test_2[index]
#
##add training points
#emsp_train_3 = np.append(emsp_train_2, emsp_train_adds, axis = 0)
#gal_Mstar_train_3 = np.append(gal_Mstar_train_2, gal_Mstar_train_adds, axis = 0)
#gal_FeH_mean_train_3 = np.append(gal_FeH_mean_train_2, gal_FeH_mean_train_adds, axis = 0)
#
##import new set of test points
#emsp_test_3 = np.load(cwd+"/samples_GAMMA/em_sample_points10000_3.npy")
#gal_Mstar_test_3 = np.load(cwd+"/samples_GAMMA/gal_Mstar_10000_3.npy")
#gal_FeH_mean_test_3 = np.load(cwd+"/samples_GAMMA/gal_FeH_mean_10000_3.npy")
#gal_FeH_std_test_3 = np.load(cwd+"/samples_GAMMA/gal_FeH_std_10000_3.npy")
#
##=================================
##Third generation of the emulator
##=================================
##Stellar Mass
#sigma_train_3 = np.zeros(len(gal_Mstar_train_3[0])) 
#em_Mstar_3 = []
#log_em_Mstar_pred_3 = np.zeros([len(gal_Mstar_test_3),len(gal_Mstar_test_3[0])])
#log_em_Mstar_pred_std_3 = np.zeros([len(gal_Mstar_test_3),len(gal_Mstar_test_3[0])])
#log_gal_Mstar_train_3 = np.log10(gal_Mstar_train_3)
##print(log_gal_Mstar_train_3)
#
#print("Training stellar mass emulators")
##Stellar mass emulators
#for i in range(len(gal_Mstar_train_3[0])):
#    sigma_train_3[i] = np.log10(np.std(gal_Mstar_train_3[:,i]))
#    #print(sigma_train[i])
#    kern = C(sigma_train_3[i]**3) * RBF() * Matern() + WhiteKernel()
#    em_Mstar_3.append(GaussianProcessRegressor(kernel=kern, n_restarts_optimizer=1))
#    em_Mstar_3[i].fit(emsp_train_3,log_gal_Mstar_train_3[:,i])
#    log_em_Mstar_pred_3[:,i], log_em_Mstar_pred_std_3[:,i] = em_Mstar_3[i].predict(emsp_test_3, return_std = True)
#    print(".",end="")
#print()
#    
##Metallicity
#sigma_train_3 = np.zeros(len(gal_FeH_mean_train_3[0])) 
#em_FeH_mean_3 = []
#em_FeH_mean_pred_3 = np.zeros([len(gal_FeH_mean_test_3),len(gal_FeH_mean_test_3[0])])
#em_FeH_mean_pred_std_3 = np.zeros([len(gal_FeH_mean_test_3),len(gal_FeH_mean_test_3[0])])
#
#print("Training metallicity emulators")
##Metallicity emulators
#for i in range(len(gal_FeH_mean_train_3[0])):
#    sigma_train_3[i] = np.std(gal_FeH_mean_train_3[:,i])
#    #print(sigma_train[i])
#    kern = C(sigma_train_3[i]**2) * RBF() * PairwiseKernel()
#    em_FeH_mean_3.append(GaussianProcessRegressor(kernel=kern, n_restarts_optimizer=1))
#    em_FeH_mean_3[i].fit(emsp_train_3,gal_FeH_mean_train_3[:,i])
#    em_FeH_mean_pred_3[:,i], em_FeH_mean_pred_std_3[:,i] = em_FeH_mean_3[i].predict(emsp_test_3, return_std = True)
#    print(".", end="")
#print()
#
##refer to first generation section for use of MultiOutputRegressor class
#
print()
print("Writing second generation of emulator models to disk")
#writes the emulator models to disk after finishing model refinement
dump(em_Mstar_2, cwd+'/Emulator_results/stellar_mass_emulator.joblib')
dump(em_FeH_mean_2, cwd+'/Emulator_results/metallicity_emulator.joblib')

#to load the emulator from disk, use this syntax
#stellar_mass_em = load('stellar_mass_emulator.joblib')
#metallicity_emulator_em = load('metallicity_emulator.joblib')

print("Emulator creation and refinement total time: {:.1f}".format(time.time()-start))
