

################################
####     Simulation 2       ####
################################

# =============================================================================
# Setup
# =============================================================================
import random
from scipy.stats import uniform
import numpy as np
import bet.calculateP.calculateR as calculateR
import bet.sample as samp
import bet.sampling.basicSampling as bsam
import matplotlib.pyplot as plt
# Python script containing SIR model
from Sim1_Model import my_SIR_model2
from Sim1_Model import my_SIR_solutions  # s, i, r curves 

# Get Initial Conditions
import Sim2_Initial_Conditions as init_cond
num_days= init_cond.num_days
t = np.linspace(0, num_days, num_days) 
S0 = init_cond.S0
T0 = init_cond.T0


num_samples = 10000
num_samples_obs = 10000

# Set "true" domain to generate data; try to reobtain domain
random.seed(3)
np.random.seed(3)

# -----------------------------------
# Beta = infection rate (parameter 0)
# Gamma = recovery rate (parameter 1)
# -----------------------------------

beta1_true = 0
beta2_true = 1
gamma1_true = 0
gamma2_true = 1


lambda_true = np.array([[beta1_true, beta2_true],
                        [gamma1_true, gamma2_true]])


# -----------------------------------------------------------------------------
# Generate observed data ------------------------------------------------------
# -----------------------------------------------------------------------------

def solveSIPSim2(beta1, beta2, gamma1, gamma2, T): 
    # T1 = T0 + T
    sampler_obs = bsam.sampler(my_SIR_model2) # Sampler for "true" QoI
    input_samples_obs = samp.sample_set(2)
    input_samples_obs.set_domain(lambda_true)
    
    # Generate samples on the parameter space
    
    domain_obs = input_samples_obs.get_domain()
    
    # Data-generating beta distributions
    input_samples_obs = sampler_obs.random_sample_set(
        # [['beta', {'a': 3, 'b': 15}], 
        [['beta', {'a': 12, 'b': 30}], 
          # ['beta', {'a': 1, 'b': 15}]], 
            ['beta', {'a': 6, 'b': 30}]], 
        input_samples_obs, num_samples=num_samples_obs)
    
    s_mat, i_mat, r_mat = my_SIR_solutions(input_samples_obs.get_values())
    q_maT = i_mat
    q_initial = (q_maT[:, T0 + T - 1] - q_maT[:, T0 - 1])/T # "Observed" data at intial time; use to solve inverse
    
    output_samples_obs = samp.sample_set(1)
    output_samples_obs.set_values(q_initial)
    
    # -----------------------------------------------------------------------------
    # Sample parameter domain uniformly, compute QoIs -----------------------------
    # -----------------------------------------------------------------------------
    
    sampler = bsam.sampler(my_SIR_model2)
    
    # -----------------------------------------------------------------------------
    # Solve SIP
    # -----------------------------------------------------------------------------

    # Initialize 2-dimensional input parameter sample set object
    input_samples = samp.sample_set(2)

    # Set parameter domain - Determine reasonable values for beta and gamma
    input_samples.set_domain(np.array([[beta1, beta2],    # beta
                                    [gamma1, gamma2]])) # gamma
    
    
    # Generate uniform samples on the parameter space
    input_samples = sampler.random_sample_set('uniform', input_samples, num_samples=num_samples)
    
    output_samples_values = my_SIR_model2(input_samples.get_values(), T = T)
    output_samples = samp.sample_set(1)
    output_samples.set_values(output_samples_values)
    
    # Create the prediction discretization object using the input samples
    disc_predict = samp.discretization(input_samples, output_samples)
    
    # # Set probability set for predictions
    disc_predict.set_output_observed_set(output_samples_obs)
    # Inversion
    calculateR.invert_to_kde(disc_predict)
       
    input_sample_values = input_samples.get_values() 
    weights = input_samples.get_weights()

# def plotContours(input_sample_values, weights, S0, beta1, beta2, gamma1, gamma2, QoI = "QoI", T = "T", weights_as_density = False):
    
    plt.rcParams.update({'font.size': 18})
    # Plot red line gamma = S0*beta  
    start = max(gamma1, beta1)
    end = min(gamma2, beta2)
     
    # Adjust weights to reflect density - weights need to be multiplied by 
    #   initial uniform sampling density
    
    plotting_weights = weights
    # if weights_as_density:
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
    plt.title("Qi(" + str(T) + ") Contours", size = 13)
    plt.gca().set_xlabel(r'$\beta$')
    plt.gca().set_ylabel(r'$\gamma$')
    plt.xlim(beta1, beta2)
    plt.ylim(gamma1, gamma2)
    plt.colorbar(scatter, label = "density")
    plt.tight_layout()
    plt.show() 
    
    return True
    

solveSIPSim2(beta1 = 0.15, beta2 = 0.45, gamma1 = 0.05, gamma2 = 0.3, T = 30)
solveSIPSim2(beta1 = 0.15, beta2 = 0.45, gamma1 = 0.05, gamma2 = 0.3, T= 31)
solveSIPSim2(beta1 = 0.15, beta2 = 0.45, gamma1 = 0.05, gamma2 = 0.3, T= 64)