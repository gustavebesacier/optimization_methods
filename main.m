%% Exercise 1
S = 100; %Current value of the stock, S(0) 
K = 100; %Strike of the option
T = 1.5; %Expiration
r = 0.04; %Risk-free rate
sigma_0 = 0.5; %Chosen at random in interval (O,1)
epsilon = 1e-12; %Tolerance 
[sigma, iterations] = newtonmethod1D(sigma_0, T, S, r, K, epsilon)

%% Exercise 2a
x = [3;3];
y = [2;2];
[x_star, j] = newtonmethod2D(3, 3, epsilon)
[y_star, j] = newtonmethod2D(2,2, epsilon)

%% Exercise 2b
grad_f = @(x) [4*( ...
    x(1) - 2)^3 + 2*(x(1) - 2*x(2)); -4*(x(1) - 2*x(2))];
[x_gd_star, k] = cstgradientdescent(grad_f, x, epsilon);
%[y_gd_star, k] = cstgradientdescent(grad_f, y, epsilon);
