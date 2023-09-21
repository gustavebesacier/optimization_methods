function [x_star, j] = newtonmethod2D(x_0, y_0, epsilon)

f = @(x) (x(1) - 2)^4 + (x(1)-2*x(2))^2;
grad_f = @(x) [4*(x(1) - 2)^3 + 2*(x(1) - 2*x(2)); -4*(x(1) - 2*x(2))];
hess_f = @(x) [12*(x(1) - 2)^2 + 2, -4 ; -4, 8 ];

x_star = [x_0; y_0];
norme_grad = norm(grad_f(x_star), 1);

j=0;

while norme_grad > epsilon
    x_star = x_star - (feval(hess_f, x_star))\feval(grad_f, x_star);
    norme_grad = norm(grad_f(x_star));
    j = j+1;
end

end

