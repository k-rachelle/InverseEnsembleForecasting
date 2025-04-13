
T = 30

I0, R0 = 0.001, 0 # I0 chosen according to dist. of case_prop(j) at time T0(j)
#                     # for county j   

# Everyone else, S0, is susceptible to infection initially.
S0 = 1 - I0 - R0

# Initial conditions vector (proportions)
y0 = S0, I0, R0


num_days = 100 # Originally 80


T0 = 10 # Start of surge
T1 = T0 + T # End of surge


