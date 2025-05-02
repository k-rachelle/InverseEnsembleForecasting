
################################
####     Simulation 4       ####
################################

# -----------------------------------------------------------------------------
# Setup -----------------------------------------------------------------------
# -----------------------------------------------------------------------------

import random
from scipy.stats import uniform
import numpy as np
import bet.calculateP.calculateR as calculateR
import bet.sample as samp
import bet.sampling.basicSampling as bsam
import matplotlib.pyplot as plt

# Python script containing SIR model
import Sim3_Model as model
t = model.t
S0 = model.S0
my_SIR_solutions = model.my_SIR_solutions
my_SIR_model = model.my_SIR_model
my_SIR_model_2D = model.my_SIR_model_2D

# -----------------------------------------------------------------------------
# Initial Conditions ----------------------------------------------------------
# -----------------------------------------------------------------------------

# Solve with firstQoI_initial(firstT_initial), secondQoI_initial(secondT_initial)

# Set Initial Conditions
T0 = 10

num_samples = 10000
num_samples_obs = 10000
num_resamples = 10000


# -----------------------------------------------------------------------------
# Domain ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Set "true" domain to generate data; try to reobtain domain
random.seed(3)
np.random.seed(3)
# -----------------------------------
# Beta = infection rate (parameter 0)
# Gamma = recovery rate (parameter 1)
# -----------------------------------

beta1_true = 0
beta2_true = 0.35
gamma1_true = 0
gamma2_true = 0.6


lambda_true = np.array([[beta1_true, beta2_true],
                        [gamma1_true, gamma2_true]])


# -----------------------------------------------------------------------------
# Generate observed data ------------------------------------------------------
# -----------------------------------------------------------------------------
sampler_obs = bsam.sampler(my_SIR_model_2D) # Sampler for "true" QoI
input_samples_obs = samp.sample_set(2)
input_samples_obs.set_domain(lambda_true)

# Generate samples on the parameter space

domain_obs = input_samples_obs.get_domain()

# Data-generating beta distributions
input_samples_obs = sampler_obs.random_sample_set(
    [['beta', {'a': 12, 'b': 30}], 
        ['beta', {'a': 6, 'b': 30}]], 
    input_samples_obs, num_samples=num_samples_obs)

s_mat, i_mat, r_mat = my_SIR_solutions(input_samples_obs.get_values())

def solveSIPSim3(beta1, beta2, gamma1, gamma2, firstT_initial, firstQoI_initial, secondT_initial, secondQoI_initial):

    QoI_lab = "[Q" + firstQoI_initial + "," + secondQoI_initial + "(" + str(firstT_initial) + ", " + str(secondT_initial) + ")]" 
  
    # -------------------------------------------------------------------------
    # Get Initial and Final QoI------------------------------------------------
    # -------------------------------------------------------------------------
    if firstQoI_initial == "s":
        firstq_mat_initial = 1-s_mat
    elif firstQoI_initial == "i":
        firstq_mat_initial = i_mat
        
    if secondQoI_initial == "s":
        secondq_mat_initial = 1-s_mat
    elif secondQoI_initial == "i":
        secondq_mat_initial = i_mat
        
    # if QoI_final == "s":
    #     q_mat_final = 1-s_mat
    # elif QoI_final == "i":
    #     q_mat_final = i_mat
    # elif QoI_final == "r":
    #     q_mat_final = r_mat
    
    firstq_initial = (firstq_mat_initial[:, T0 + firstT_initial - 1] - firstq_mat_initial[:, T0 - 1])/firstT_initial # "Observed" data at initial time; use to solve inverse
    secondq_initial = (secondq_mat_initial[:, T0 + secondT_initial - 1] - secondq_mat_initial[:, T0 - 1])/secondT_initial # "Observed" data at initial time; use to solve inverse

    output_samples_obs = samp.sample_set(2)   # Changed to (2) instead of (1) for 2D QoI output
    output_samples_obs.set_values([firstq_initial, secondq_initial])
    
    # -------------------------------------------------------------------------
    # Sample parameter domain uniformly, compute QoIs -------------------------
    # -------------------------------------------------------------------------
    
    sampler = bsam.sampler(my_SIR_model_2D)
    
    
    # Initialize 2-dimensional input parameter sample set object
    input_samples = samp.sample_set(2)

    # Set parameter domain - Determine reasonable values for beta and gamma
    input_samples.set_domain(np.array([[beta1, beta2], 
                                       [gamma1, gamma2]])) 
    
    
    # Generate uniform samples on the parameter space
    input_samples = sampler.random_sample_set('uniform', input_samples, num_samples=num_samples)
    
    output_samples_values = my_SIR_model_2D(input_samples.get_values(), T0 = T0,  firstT = firstT_initial, secondT = secondT_initial, Q1 = firstQoI_initial, Q2 = secondQoI_initial)
    output_samples = samp.sample_set(2)
    output_samples.set_values(output_samples_values)
    
    # Create the prediction discretization object using the input samples
    disc_predict = samp.discretization(input_samples, output_samples)
    
    # # Set probability set for predictions
    disc_predict.set_output_observed_set(output_samples_obs)
    
    
    # Inversion
    calculateR.invert_to_kde(disc_predict)
    
       
    input_sample_values = input_samples.get_values() 
    weights = input_samples.get_weights()
    
    plt.rcParams.update({'font.size': 18})
    # Plot red line gamma = S0*beta  
    start = max(gamma1, beta1)
    end = min(gamma2, beta2)
     
    # Adjust weights to reflect density - weights need to be multiplied by 
    #   initial uniform sampling density
    
    plotting_weights = weights

    for i in range(len(weights)):
        input_sample_value = input_sample_values[i]
        beta_val  = input_sample_value[0]
        gamma_val = input_sample_value[1]
        
        dens_beta = uniform.pdf(beta_val, beta1, beta2-beta1)
        dens_gamma = uniform.pdf(gamma_val, gamma1, gamma2-gamma1)
        
        plotting_weights[i] = weights[i] * dens_beta * dens_gamma
        

    scatter = plt.scatter(input_sample_values[:,0], input_sample_values[:,1],
                c=plotting_weights, s= 30)  
    plt.plot([start, end], [start * S0, end * S0], 'k-', color = 'r')
    plt.title(QoI_lab)
    plt.gca().set_xlabel(r'$\beta$')
    plt.gca().set_ylabel(r'$\gamma$')
    plt.xlim(beta1, beta2)
    plt.ylim(gamma1, gamma2)
    plt.colorbar(scatter, label = "density")
    plt.tight_layout()
    plt.show() 
    
    return True

solveSIPSim3(beta1 = 0.15, beta2 = 0.45, gamma1 = 0.05, gamma2 = 0.3, firstT_initial = 30, firstQoI_initial= "s", secondT_initial = 30, secondQoI_initial = "i")
solveSIPSim3(beta1 = 0.15, beta2 = 0.45, gamma1 = 0.05, gamma2 = 0.3, firstT_initial = 30, firstQoI_initial= "s", secondT_initial = 31, secondQoI_initial = "s")
