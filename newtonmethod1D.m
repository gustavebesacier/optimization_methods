function [sigma_star, iter] = newtonmethod1D(d1, d2, g, g_prime, sigma_0, T, S, r, K, epsilon)

sigma_star = sigma_0;

iter = 0;

while abs(g(sigma_star))>epsilon
    iter = iter + 1;
    sigma_star = sigma_star - g(sigma_star)/g_prime(sigma_star);
end

end
