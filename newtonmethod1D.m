function [sigma_star, iter] = newtonmethod1D(sigma_0, T, S, r, K, epsilon)

d1 = @(sigma) (log(S/K) + (r+0.5*sigma^2)*T)/(sigma*sqrt(T));
d2 = @(sigma) d1(sigma) - sigma * sqrt(T);

g = @(sigma) S*normcdf(d1(sigma)) - K*exp(-r*T)*normcdf(d2(sigma)) - 10.78;

g_prime = @(sigma) S*((exp(-d1(sigma)^2/2))/(sqrt(2*pi)))*sqrt(T);

sigma_star = sigma_0;

iter = 0;

while abs(g(sigma_star))>epsilon
    iter = iter + 1;
    sigma_star = sigma_star - g(sigma_star)/g_prime(sigma_star);
end

end
