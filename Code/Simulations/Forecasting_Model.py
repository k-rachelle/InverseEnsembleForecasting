
#############################
#### SIR Model Inversion ####
#############################

# =============================================================================
# Setup
# =============================================================================

import numpy as np
from scipy.integrate import odeint


I0, R0 = 0.001, 0 # I0 chosen according to dist. of case_prop(j) at time T0(j)
#                     # for county j   

# Everyone else, S0, is susceptible to infection initially.
S0 = 1 - I0 - R0

# Initial conditions vector (proportions)
y0 = S0, I0, R0


num_days = 100 

# Initial proportion of infected and recovered individuals, I0 and R0.
t = np.linspace(0, num_days, num_days) 


# T0 = init_cond.T0
# T1 = init_cond.T1
    
# =============================================================================
# Model
# =============================================================================

# The SIR model differential equations.
def deriv(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I  - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# SIR solutions - For plotting infection curves
def my_SIR_solutions(parameter_samples):
        if parameter_samples.shape == (2,):
            beta = parameter_samples[0]
            gamma = parameter_samples[1]
        else:
            beta = parameter_samples[:, 0]
            gamma = parameter_samples[:, 1]
            
        s_mat = np.empty([parameter_samples.shape[0], num_days])
        i_mat = np.empty([parameter_samples.shape[0], num_days])
        r_mat = np.empty([parameter_samples.shape[0], num_days])
        
        for i in range(parameter_samples.shape[0]):
            sol = odeint(deriv, y0, t, args=(beta[i], gamma[i]))
            S, I, R = sol.T

            s_mat[i] = S
            i_mat[i] = I
            r_mat[i] = R

        
        return s_mat, i_mat, r_mat


# QoI map
def my_SIR_model(parameter_samples, T0, T, Q = "s"):
        T1 = T0 + T 
        if parameter_samples.shape == (2,):
            beta = parameter_samples[0]
            gamma = parameter_samples[1]
        else:
            beta = parameter_samples[:, 0]
            gamma = parameter_samples[:, 1]
            
        QoI = np.empty(parameter_samples.shape[0])
        
        s_mat = np.empty([parameter_samples.shape[0], num_days])
        i_mat = np.empty([parameter_samples.shape[0], num_days])
        r_mat = np.empty([parameter_samples.shape[0], num_days])
        
        for i in range(parameter_samples.shape[0]):
            sol = odeint(deriv, y0, t, args=(beta[i], gamma[i]))
            S, I, R = sol.T

            s_mat[i] = S
            i_mat[i] = I
            r_mat[i] = R
            
            
            if Q == "s":   
                QoI[i] = (S[T0-1] - S[T1-1])/T
            elif Q == "i":
                QoI[i] = (I[T1] - I[T0])/T
            elif Q == "r":
                QoI[i] = (R[T1] - R[T0])/T
        return QoI



# 2D QoI map
def my_SIR_model_2D(parameter_samples, T0, firstT, secondT, Q1 = "s", Q2 = "s"):
    # Output 2D QoI; using two time points or two QoI
        firstT1 = T0 + firstT 
        secondT1 = T0 + secondT
        if parameter_samples.shape == (2,):
            beta = parameter_samples[0]
            gamma = parameter_samples[1]
        else:
            beta = parameter_samples[:, 0]
            gamma = parameter_samples[:, 1]
            
        QoI1 = np.empty(parameter_samples.shape[0])
        QoI2 = np.empty(parameter_samples.shape[0])
        
        s_mat = np.empty([parameter_samples.shape[0], num_days])
        i_mat = np.empty([parameter_samples.shape[0], num_days])
        r_mat = np.empty([parameter_samples.shape[0], num_days])
        
        for i in range(parameter_samples.shape[0]):
            sol = odeint(deriv, y0, t, args=(beta[i], gamma[i]))
            S, I, R = sol.T

            s_mat[i] = S
            i_mat[i] = I
            r_mat[i] = R
            
            
            if Q1 == "s":   
                QoI1[i] = (S[T0-1] - S[firstT1-1])/firstT
            elif Q1 == "i":
                QoI1[i] = (I[firstT1] - I[T0])/firstT
            elif Q1 == "r":
                QoI1[i] = (R[firstT1] - R[T0])/firstT
                            
            if Q2 == "s":   
                QoI2[i] = (S[T0-1] - S[secondT1-1])/secondT
            elif Q2 == "i":
                QoI2[i] = (I[secondT1] - I[T0])/secondT
            elif Q1 == "r":
                QoI2[i] = (R[secondT1] - R[T0])/secondT
        return QoI1, QoI2
