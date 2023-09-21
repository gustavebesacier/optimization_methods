function [x_star, k] = cstgradientdescent(fhandle, x0, epsilon)
    alpha = 1;
    % initialization of the algorithm
    x_star=x0;
    norme_grad = norm(fhandle(x0), 1)
    k = 0;
    while norme_grad > epsilon 
        k = 1 + k;
        grad = fhandle(x_star)
        x_star = x_star - alpha*grad
    end

end