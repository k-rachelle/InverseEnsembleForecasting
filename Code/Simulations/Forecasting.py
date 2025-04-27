
# Forecast from initial solution to a later time.


# -----------------------------------------------------------------------------
# Setup -----------------------------------------------------------------------
# -----------------------------------------------------------------------------

import random
import numpy as np
import bet.calculateP.calculateR as calculateR
import bet.sample as samp
import bet.sampling.basicSampling as bsam

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
# Python script containing SIR model
import Forecasting_Model
t = Forecasting_Model.t
S0 = Forecasting_Model.S0
my_SIR_solutions = Forecasting_Model.my_SIR_solutions
my_SIR_model = Forecasting_Model.my_SIR_model

random.seed(3)
np.random.seed(3)

###############
#### PLOTS ####
###############

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

def forecastSingleQoI(T_initial, QoI_initial, T_final, QoI_final, pred_q_linspace, l1=None, l2=None):
    # l1 and l2: bounds for shading under forecasted QoI distribution
    # Solve with QoI_initial(T_initial)
    # Forecast to QoI_final(T_final)

    # Set Initial Conditions
    T0 = 10
    
    num_samples = 10000
    num_samples_obs = 10000
    num_resamples = 10000
    
    QoI_lab = QoI_initial + "(" + str(T_initial) + ") to " + QoI_final + "(" + str(T_final) + ")"
    

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
    
    sampler_obs = bsam.sampler(my_SIR_model) # Sampler for "true" QoI
    input_samples_obs = samp.sample_set(2)
    input_samples_obs.set_domain(lambda_true)
    
    # Generate samples on the parameter space
    # Data-generating beta distributions
    
    input_samples_obs = sampler_obs.random_sample_set(
        [['beta', {'a': 12, 'b': 30}], 
            ['beta', {'a': 6, 'b': 30}]], 
        input_samples_obs, num_samples=num_samples_obs)
    
    s_mat, i_mat, r_mat = my_SIR_solutions(input_samples_obs.get_values())
    
    
    # -------------------------------------------------------------------------
    # Get Initial and Final QoI------------------------------------------------
    # -------------------------------------------------------------------------
    if QoI_initial == "s":
        q_mat_initial = 1-s_mat
    elif QoI_initial == "i":
        q_mat_initial = i_mat
        
    if QoI_final == "s":
        q_mat_final = 1-s_mat
    elif QoI_final == "i":
        q_mat_final = i_mat
    
    q_initial = (q_mat_initial[:, T0 + T_initial - 1] - q_mat_initial[:, T0 - 1])/T_initial # "Observed" data at intial time; use to solve inverse
    q_final = (q_mat_final[:, T0 + T_final - 1] - q_mat_final[:, T0 - 1])/T_final 
    
    output_samples_obs = samp.sample_set(1)
    output_samples_obs.set_values(q_initial)
    
    # -----------------------------------------------------------------------------
    # Sample parameter domain uniformly, compute QoIs -----------------------------
    # -----------------------------------------------------------------------------
    
    sampler = bsam.sampler(my_SIR_model)
    
    
    # Initialize 2-dimensional input parameter sample set object
    input_samples = samp.sample_set(2)
    
    beta1 = 0.15
    beta2 = 0.45
    gamma1 = 0.05
    gamma2 = 0.3
    
    
    
    # Set parameter domain - Determine reasonable values for beta and gamma
    input_samples.set_domain(np.array([[beta1, beta2],    # beta
                                       [gamma1, gamma2]])) # gamma
    
    
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
    joint_kde = gaussian_kde(input_sample_values.T, weights = weights)

    new_samp = joint_kde.resample(num_resamples).T
    # beta_samp = new_samp[:, 0]
    # gamma_samp = new_samp[:, 1]

    # Predict to final time
    pred_q_final = my_SIR_model(new_samp, T0, T_final, QoI_final)

    # Fit output to KDE with the weights.
    
    # Predict to final time - map input samples to the prediction time QoI
    # pred_q_final2 = my_SIR_model(input_sample_values, T0, T_final, QoI_final)
    
    # Fit KDE to prediction QoI distribution with solved weights
    pred_q_final_KDE = gaussian_kde(pred_q_final)

    plotPredictedQoIRegion(q_final, pred_q_linspace, pred_q_final_KDE, l1, l2, T_final, QoI_lab = QoI_lab)


forecastSingleQoI(T_initial = 10, QoI_initial = "s", T_final = 30, QoI_final = "s", 
                  pred_q_linspace = np.linspace(-0.005, 0.035, 5000) , l1 = -0.005, l2 = 0.02925) # l1 and l2 determined to represent a 95% probability region using findHighProbRegion()
forecastSingleQoI(T_initial = 10, QoI_initial = "s", T_final = 60, QoI_final = "s",
                  pred_q_linspace = np.linspace(-0.003, 0.021, 5000), l1 = -0.00129573, l2 = 0.01673326)
forecastSingleQoI(T_initial = 10, QoI_initial = "i", T_final = 30, QoI_final = "i",
                  pred_q_linspace = np.linspace(-0.003, 0.021, 5000), l1 = -0.00125452,  l2 = 0.01009466)
forecastSingleQoI(T_initial = 10, QoI_initial = "i", T_final = 60, QoI_final = "i",
                  pred_q_linspace = np.linspace(-0.001, 0.006, 5000), l1 = -0.00042858,  l2 = 0.0026661)

forecastSingleQoI(T_initial = 30, QoI_initial = "s", T_final = 60, QoI_final = "s",
                  pred_q_linspace = np.linspace(-0.003, 0.021, 5000), l1 = -0.00114245,  l2 = 0.016845)
forecastSingleQoI(T_initial = 30, QoI_initial = "i", T_final = 60, QoI_final = "i",
                    pred_q_linspace = np.linspace(-0.001, 0.006, 5000), l1 = -0.00047999,  l2 = 0.00250828)

forecastSingleQoI(T_initial = 30, QoI_initial = "s", T_final = 30, QoI_final = "i",
                    pred_q_linspace = np.linspace(-0.003, 0.021, 5000), l1 = -0.00122796,  l2 = 0.01062207)
forecastSingleQoI(T_initial = 30, QoI_initial = "s", T_final = 60, QoI_final = "i",
                    pred_q_linspace = np.linspace(-0.001, 0.006, 5000), l1 = -0.00045927,  l2 = 0.00270688)

