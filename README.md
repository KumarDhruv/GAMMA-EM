# GAMMA-EM
Gaussian Process-based Emulators of the GAMMA model for Galactic Chemical Evolution

# Description
GAMMA-EM is a pipeline for comparison of the GAMMA model with observational data taken from the APOGEE survey and McConnachie 2012. It utilizes Gaussian Process regression to create model emulators for GAMMA, then uses the emcee package to run Markov Chain Monte Carlo simulations to explore the parameter space.

# Requirements  
* Python 3.7
* [NuPyCEE](https://github.com/NuGrid/NuPyCEE)   
* [JINAPyCEE](https://github.com/becot85/JINAPyCEE/tree/master_carleen)  
* PyDOE 
* emcee v2.2.x
* HPCC  

# How to use
In DOCS, please see the GAMMA-EM_MCMC-Walkthrough.ipynb notebooks for a tutorial on how to walk through these files to produce model emulators and MCMC results
