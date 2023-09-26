function [x_star, k] = cstgradientdescent(fhandle, x0, alpha, epsilon)
    
    % initialization of the algorithm
    x_star = x0;
    norme_grad = norm(fhandle(x0), Inf);
    k = 0;
    while norme_grad > epsilon 
        k = 1 + k;
        grad = fhandle(x_star);
        x_star = x_star - alpha*grad;
        norme_grad = norm(fhandle(x_star),Inf);
    end
end
