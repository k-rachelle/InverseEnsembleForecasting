
#### SIR Model Inversion ####
# Setup

import numpy as np
from scipy.integrate import odeint
import Sim1_Initial_Conditions as init_cond

T = init_cond.T
y0 = init_cond.y0

# Initial proportion of infected and recovered individuals, I0 and R0.


num_days = init_cond.num_days
t = np.linspace(0, num_days, num_days) 


T0 = init_cond.T0
T1 = init_cond.T1
    
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


# QoI map (1-S)
def my_SIR_model(parameter_samples):
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
            
            
            QoI[i] = (S[T0-1] - S[T1-1])/T

        
        return QoI


# QoI map 2 (I)
def my_SIR_model2(parameter_samples):
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
            
            
            QoI[i] = (I[T1-1] - I[T0-1])/T
            # QoI[i] = (I[T1-1])

        
        return QoI

def my_SIR_model3(parameter_samples):
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
            
            
            QoI[i] = (R[T1-1] - R[T0-1])/T
            # QoI[i] = (I[T1-1])

        
        return QoI



