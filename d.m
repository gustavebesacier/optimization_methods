function [d1, d1_prime] = d (S,K,T,r)
d1 = @(sigma) (log(S/K) + (r+0.5*sigma*sigma)*T)/(sigma*sqrt(T));
d2 = d1 - sigma * sqrt(T)
d1_prime = @(sigma) sqrt(T) - log(S/K)/(sigma*sigma*sqrt(T)) - ((r+0.5*sigma*sigma)*sqrt(T))/(sigma*sigma);