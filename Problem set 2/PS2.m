clear

% Initializing
x1 = [1,1];
x2 = [2,2];
x3 = [3,3];

% Parameters
alpha = 1;
beta = 0.5;
gamma = 2;
epsilon = 1e-6;

% Rosenbrock function
f = @(x) (1-x(1))^2+10*(x(2)-x(1)^2)^2;
fx = [f(x1); f(x2); f(x3)];
diff = max(fx) - min(fx);

while diff > epsilon
    res = 0;
    simplex = [x1',x2',x3']; 
    k = 0;
    for i = 1:length(simplex)
        if fx(i) > res
            res = fx(i);
            worse = simplex(i);
            i = k;
        end
    end 
    simplex(k) = [];

    c = [(simplex(1)+simplex(3))/2; 
        (simplex(2)+simplex(4))/2]; 
    x_try = (1+alpha)*c -alpha*c;
    if f(x_try) > f(worse)
        x_new = (1 - beta)*c + beta*worse;
    else 
        x_new = (1 + gamma)*x_try - gamma*c;
    end
    
    simplex = [simplex, x_new];
    fx(k) = f(x_new);
    diff = max(fx) - min(fx);
end