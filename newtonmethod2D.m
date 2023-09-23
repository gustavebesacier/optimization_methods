function [x_star, j] = newtonmethod2D(f, grad_f, hess_f, x, epsilon)

x_star = x;
norme_grad = norm(grad_f(x_star), 1);

j=0;

while norme_grad > epsilon
    x_star = x_star - (feval(hess_f, x_star))\feval(grad_f, x_star);
    norme_grad = norm(grad_f(x_star));
    j = j+1;
end

end
