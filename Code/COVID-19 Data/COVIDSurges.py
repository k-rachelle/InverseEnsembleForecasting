
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
import COVIDSurges_Model as model
t = model.t
S0 = model.S0
my_SIR_solutions = model.my_SIR_solutions
my_SIR_model = model.my_SIR_model

random.seed(1)
np.random.seed(1)

# -----------------------------------------------------------------------------
# Initial Conditions ----------------------------------------------------------
# -----------------------------------------------------------------------------

###############
#### PLOTS ####
###############


# =============================================================================
# Plot contours
# =============================================================================

def plotContours(input_sample_values, weights, S0, beta1, beta2, gamma1, gamma2, QoI = "QoI", T = "T", weights_as_density = False):
    # Plot red line gamma = S0*beta
    plt.rcParams.update({'font.size': 18})    
    start = max(gamma1, beta1)
    end = min(gamma2, beta2)
    
    plotting_weights = weights
    if weights_as_density:
        for i in range(len(weights)):
            input_sample_value = input_sample_values[i]
            beta_val  = input_sample_value[0]
            gamma_val = input_sample_value[1]
            
            dens_beta = uniform.pdf(beta_val, beta1, beta2-beta1)
            dens_gamma = uniform.pdf(gamma_val, gamma1, gamma2-gamma1)
            
            plotting_weights[i] = weights[i] * dens_beta * dens_gamma
            
    scatter = plt.scatter(input_sample_values[:,0], input_sample_values[:,1],
                c=plotting_weights)
    plt.plot([start, end], [start * S0, end * S0], 'k-', color = 'r')
    plt.gca().set_xlabel(r'$\beta$')
    plt.gca().set_ylabel(r'$\gamma$')
    plt.title("Q" + QoI + "(" + str(T) + ") Contours")
    plt.colorbar(scatter, label = "density")
    plt.tight_layout()

    plt.show()

# =============================================================================
# Plot predicted with observed QoIs
# =============================================================================

def plotPredictedQoIRegion(output_obs, pred_q_linspace, KDE, l1=None, l2=None, T = "T", title = "", QoI_lab = ""):

    pred_q_dens = KDE.evaluate(pred_q_linspace)
    plt.hist(output_obs, rwidth = 0.93, alpha = 0.5, density = True, bins = 20, label = "Observed")
    plt.plot(pred_q_linspace, pred_q_dens) 
    plt.title(title + QoI_lab)
    plt.legend()
    if l1 is None or l2 is None:
        plt.show()
    else:
        plt.fill_between(pred_q_linspace, pred_q_dens,
                         where = (pred_q_linspace > l1) & (pred_q_linspace < l2),
                         alpha = 0.5,
                          color = "orange")

        plt.show()
        
    
def bisectionSearch(x0, x1, KDE, linspace, L):

    x2 = x0
    while abs(x0 - x1) > 0.00001:
        x2 = (x0 + x1)/2 
        
        # Check if f(xi) - L and f(xi-1)- L have different signs
        f_x2 = KDE.evaluate(x2)
        f_x0 = KDE.evaluate(x0)

        if (f_x2 - L < 0 and f_x0 - L > 0) or  (f_x2 - L > 0 and f_x0 - L < 0):
            x1 = x2
        else:
            x0 = x2
        
    return x2
     
# Multimodal KDEs could cause problems
def findHighProbRegion(KDE, # KDE fit of interest
                       linspace,  # Seqence of values to consider the KDE density
                       L_max,
                       L_min = 0,
                       p = 0.95, # Desired area of region
                       thres = 0.001,
                       linspace_len = 10000
                       ):

    # Find M := max density over linspace
    dens = KDE.evaluate(linspace)
    M = max(dens)
    m = linspace[dens == M]
    if L_max is None: 
        L_max = M
    min_x = linspace[0]
    max_x = linspace[linspace.size - 1]
    
    for L in np.linspace(L_min, L_max, linspace_len):
        print(L)
        # Find Points for which density = L
        l1 = bisectionSearch(min_x, m, KDE, linspace, L)
        l2 = bisectionSearch(m, max_x, KDE, linspace, L)
        
        # Calculate area between l1 and l2
        A = KDE.integrate_box_1d(min_x, l2) - KDE.integrate_box_1d(min_x, l1)
        print("A: " + str(A))
        print("")
        # If area is close to p, end. Else lower L.
        if abs(A - p) < thres:
            break
        else:
            continue
    
    return l1, l2, A, L


def solveSIPSurge(beta1, beta2, gamma1, gamma2, T_initial, q_initial):
    QoI_initial = "s"  # Must be "s"
    # Set Initial Conditions
    T0 = 1
    
    num_samples = 10000
    
    # QoI_lab = "Qs(" + str(T_initial) + ")"
    
    
    # -----------------------------------------------------------------------------
    # Domain ----------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    
    # Set "true" domain to generate data; try to reobtain domain
    random.seed(3)
    
    output_samples_obs = samp.sample_set(1)
    output_samples_obs.set_values(q_initial)
    
    # -------------------------------------------------------------------------
    # Sample parameter domain uniformly, compute QoIs -------------------------
    # -------------------------------------------------------------------------
    
    sampler = bsam.sampler(my_SIR_model)
    
    
    # Initialize 2-dimensional input parameter sample set object
    input_samples = samp.sample_set(2)
    
    input_samples.set_domain(np.array([[beta1, beta2],
                                       [gamma1, gamma2]])) 
    
    
    # Generate uniform samples on the parameter space
    input_samples = sampler.random_sample_set('uniform', input_samples, num_samples=num_samples)
    
    output_samples_values = my_SIR_model(input_samples.get_values(), T0 = T0,  T = T_initial, Q = QoI_initial)
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
    
    plotContours(input_sample_values, weights, S0, beta1, beta2, gamma1, gamma2, QoI = QoI_initial, T = T_initial, weights_as_density=True)
# 
    
# Solve with Qs(T_initial)
# Forecast to Qs(T_final)
def forecastSIPSurge(beta1, beta2, gamma1, gamma2, T_initial, T_final, q_initial, q_final, pred_q_linspace, l1, l2):
    QoI_initial = "s"  # Must be "s"
    # Set Initial Conditions
    T0 = 1
    
    num_samples = 10000
    num_resamples = 10000
    
    QoI_lab = "Qs(" + str(T_initial) + ") to " + "Qs(" + str(T_final) + ")"
    
    
    # -----------------------------------------------------------------------------
    # Domain ----------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    
    # Set "true" domain to generate data; try to reobtain domain
    random.seed(3)
    
    output_samples_obs = samp.sample_set(1)
    output_samples_obs.set_values(q_initial)
    
    # -------------------------------------------------------------------------
    # Sample parameter domain uniformly, compute QoIs -------------------------
    # -------------------------------------------------------------------------
    
    sampler = bsam.sampler(my_SIR_model)
    
    
    # Initialize 2-dimensional input parameter sample set object
    input_samples = samp.sample_set(2)
    
    input_samples.set_domain(np.array([[beta1, beta2],
                                       [gamma1, gamma2]])) 
    
    
    # Generate uniform samples on the parameter space
    input_samples = sampler.random_sample_set('uniform', input_samples, num_samples=num_samples)
    
    output_samples_values = my_SIR_model(input_samples.get_values(), T0 = T0,  T = T_initial, Q = QoI_initial)
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
    
    # -------------------------------------------------------------------------
    # Resample from solution --------------------------------------------------
    # -------------------------------------------------------------------------
    
    from scipy.stats import gaussian_kde
    
    joint_kde = gaussian_kde(input_sample_values.T, weights = weights)
    
    new_samp = joint_kde.resample(num_resamples).T
    # beta_samp = new_samp[:, 0]
    # gamma_samp = new_samp[:, 1]

    # Predict to final time
    pred_q_final = my_SIR_model(new_samp, T0, T_final, QoI_initial)

    # Fit prediction to KDE
    pred_q_final_KDE = gaussian_kde(pred_q_final)
    
    plotPredictedQoIRegion(q_final, pred_q_linspace, pred_q_final_KDE, l1, l2, T_final,  QoI_lab = QoI_lab)


# Solve for Surge 1 with Qs(64)
q_initial = np.genfromtxt("Data/Surge1_QoI_64_days.csv", delimiter = ",", 
                              skip_header = 1) 
solveSIPSurge(beta1 = 0.05, beta2 = 0.32, gamma1 = 1/15, gamma2 = 1/4,
               T_initial = 64, q_initial = q_initial)



# Solve for Surge 2 with Qs(15)
q_initial = np.genfromtxt("Data/Surge2_QoI_15_days.csv", delimiter = ",", 
                              skip_header = 1) 
solveSIPSurge(beta1 = 0.18, beta2 = 0.5, gamma1 = 1/14, gamma2 = 1/3,
               T_initial = 15, q_initial = q_initial)


# Surge 1 forecasts------------------------------------------------------------
# QoI saved for T = 30, 64, 100

# Solve at Qs(30) and forecast to Qs(64)
q_initial = np.genfromtxt("Data/Surge1_QoI_30_days.csv", delimiter = ",", 
                              skip_header = 1)  
q_final = np.genfromtxt("Data/Surge1_QoI_64_days.csv", delimiter = ",", 
                              skip_header = 1)  
forecastSIPSurge(beta1 = 0.05, beta2 = 0.32, gamma1 = 1/15, gamma2 = 1/4,
                  T_initial = 30, T_final = 64, q_initial = q_initial, q_final = q_final,
                  pred_q_linspace = np.linspace(-0.0005, 0.004, 1000) , l1 = -0.00012975, l2 = 0.00198705) # l1 and l2 determined to represent a 95% probability region using findHighProbRegion()


# Solve at Qs(30) and forecast to Qs(100)
q_initial = np.genfromtxt("Data/Surge1_QoI_30_days.csv", delimiter = ",", 
                              skip_header = 1)  
q_final = np.genfromtxt("Data/Surge1_QoI_100_days.csv", delimiter = ",", 
                              skip_header = 1)  
forecastSIPSurge(beta1 = 0.05, beta2 = 0.32, gamma1 = 1/15, gamma2 = 1/4,
                  T_initial = 30, T_final = 100, q_initial = q_initial, q_final = q_final,
                  pred_q_linspace = np.linspace(-0.0005, 0.006, 1000)  , l1 = -0.00038633, l2 = 0.00399271)


# Surge 2 forecasts------------------------------------------------------------
# QoI saved for T = 10, 15, 30
# Solve at Qs(10) and forecast to Qs(15)
q_initial = np.genfromtxt("Data/Surge2_QoI_10_days.csv", delimiter = ",", 
                              skip_header = 1)  
q_final = np.genfromtxt("Data/Surge2_QoI_15_days.csv", delimiter = ",", 
                              skip_header = 1)  
forecastSIPSurge(beta1 = 0.18, beta2 = 0.5, gamma1 = 1/14, gamma2 = 1/3,
                  T_initial = 10, T_final = 15, q_initial = q_initial, q_final = q_final, 
                  pred_q_linspace = np.linspace(0, 0.005, 1000), l1 =  0.00030647, l2 = 0.00339903)

# Solve at Qs(10) and forecast to Qs(30)
q_initial = np.genfromtxt("Data/Surge2_QoI_10_days.csv", delimiter = ",", 
                              skip_header = 1)  
q_final = np.genfromtxt("Data/Surge2_QoI_30_days.csv", delimiter = ",", 
                              skip_header = 1)  
forecastSIPSurge(beta1 = 0.18, beta2 = 0.5, gamma1 = 1/14, gamma2 = 1/3,
                  T_initial = 10, T_final = 30, q_initial = q_initial, q_final = q_final, 
                  pred_q_linspace = np.linspace(-0.0015, 0.03, 1000), l1 =  -0.00023179, l2 = 0.01902726)
