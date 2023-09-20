S = 100; %Current value of the stock, S(0) 
K = 100; %Strike of the option
T = 1.5; %Expiration
r = 0.04; %Risk-free rate
sigma_0 = 0.5; %Chosen at random in interval (O,1)
epsilon = 1e-12; %Tolerance 
[sigma, iterations] = newtonmethod1D(sigma_0, T, S, r, K, epsilon)


