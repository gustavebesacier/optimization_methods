function [x_star, j] = newtonmethod1D(x_0, T, S, r, K, epsilon)
sigma = 1;
d_1 = (log(S/K) + (r+0.5*sigma*sigma)*T)/(sigma*sqrt(T));
d_2 = d_1 - sigma * sqrt(T);
d1 = @(sigma) d_1;
d2 = @(sigma) d_2;

f = @(sigma) S*normcdf(d1(sigma)) - K*exp(-r*T)*normcdf(d2(sigma));

f_prime = @(sigma) S*((exp(-d1(sigma)^2/2))/(sqrt(2*pi)))*sqrt(T);

x_star = x_0;

j = 0;
while abs(f(x_star))>epsilon
    j = j + 1;
    x_star = x_star - feval(f,x_star)/feval(f_prime,x_star);
    x_star
    f(x_star)
end
end