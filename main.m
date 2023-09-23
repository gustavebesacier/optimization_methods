clear
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
grad_f = @(q) [4*(q(1) - 2)^3 + 2*(q(1) - 2*q(2)); -4*(q(1) - 2*q(2))];

x = [3;3];
y = [2;2];

alpha1 = 0.1;
alpha2 = 1e-2;
alpha3 = 1e-3;

epsilon=1e-6;

[x_gd_star_1, a] = cstgradientdescent(grad_f, x, alpha1,epsilon)
[x_gd_star_2, b] = cstgradientdescent(grad_f, x, alpha2,epsilon)
[x_gd_star_3, c] = cstgradientdescent(grad_f, x, alpha3,epsilon)

[y_gd_star_1, e] = cstgradientdescent(grad_f, y, alpha1,epsilon)
[y_gd_star_2, f] = cstgradientdescent(grad_f, y, alpha2,epsilon)
[y_gd_star_3, g] = cstgradientdescent(grad_f, y, alpha3,epsilon)
