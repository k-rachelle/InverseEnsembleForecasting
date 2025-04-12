# Initial conditions for simulations
T = 30
I0, R0 = 0.001, 0

# Everyone else, S0, is susceptible to infection initially.
S0 = 1 - I0 - R0

# Initial conditions vector (proportions)
y0 = S0, I0, R0

num_days = 150

T0 = 10 # Start of surge
T1 = T0 + T # End of surge


