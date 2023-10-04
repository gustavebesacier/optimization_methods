# import packages
import numpy.linalg as l
import numpy as np

x0 = [1, 2]

mu = 10
epsilon = 10**(-10)
x_next = x_0


# Definition of the function
def func(x):
    return (x[0]-2) ** 2 + (x[0] - 2*x[1]) ** 2

# Definition of the constraint function
def constraint(x):
    return (x[0] - x[1]**2)**2

def auxiliary(x, mu):
    return func + mu*(constraint(x))

def grad_aux(x, mu):
    return l.array[(4 + 2*mu)*x[0] - 2*(2 + mu*x[1])*x[1] - 2, 
                   -2*(1 + mu*x[1])*x[0] + 4*(1 + mu*(x[1]**2))*x[1]]

def hess_aux(x, mu):
    return l.array([[4+2*mu, 4*(1+mu*x[1])], 
                     [-2*(1+mu*x[1]), -2*mu*x[0] + 4*(1+3*mu*(x[1]**2))]])

def gradient_descent(func, x, epsilon):
    x_star = x
    norme_grad = l.norm(func(x_star), np.inf)
    while norme_grad > epsilon:
        grad = func(x_star)
        x_star = x_star - 0.001*grad
        norme_grad = l.norm(func(x_star), np.inf)
    return x_star

def newton_method(func, grad_func, hess_func, x, mu, epsilon):
    x_star = x
    norme_grad = l.norm(grad_func(x_star, mu), 1)
 
    while norme_grad > epsilon:
     x_star = x_star - l.inv(hess_func(x_star, mu))*grad_func(x_star, mu)
     norme_grad = l.norm(grad_func(x_star, mu))

    return x_star

if __name__ == '__main__':


    # Loop, continues until the difference between the smallest and the largest value of the evaluation of the Rosenbrock function at the simplex points is smaller than the tolerance level
    while epsilon < diff:
    
        x_next = newton_method(auxiliary, grad_aux, hess_aux, x_next, mu, epsilon)
        mu = 1/2 * mu

        # For each point, if the evaluation of the function at this point is larger than the worst point, it becomes the worst point
        for i in range(len(simplex)):

            if fx[i] >= res:

                res = fx[i]
                worse = simplex[i]
                k = i

        # Remove the worst point from the points of the simplex
        simplex.remove(worse)

        # Centroid computation
        c = [(simplex[0][0] + simplex[1][0]) / 2, (simplex[0][1] + simplex[1][1]) / 2]

        # x_try
        x_try = [(1 + alpha) * c[0] - alpha * worse[0], (1 + alpha) * c[1] - alpha * worse[1]]

        # Contraction of the simplex, if x_try is worse than the worst point
        if rosenbrock(x_try) > rosenbrock(worse):
            x_new = [(1 - beta) * c[0] + beta * worse[0], (1 - beta) * c[1] + beta * worse[1]]

        # Expansion of the simplex, if x_try is such that f(x_try) is better than the other evaluations previously computed
        elif rosenbrock(x_try) < best:
            x_new = [(1 + gamma) * x_try[0] - gamma * c[0], (1 + gamma) * x_try[1] - gamma * c[1]]

        # No contraction / no expansion
        else:
            x_new = x_try

        # Add the new point to the simplex
        simplex.append(x_new)

        # Re-compute the value of the function at the points of the simplex
        fx = [rosenbrock(simplex[0]), rosenbrock(simplex[1]), rosenbrock(simplex[2])]

        best = min(fx)
        diff = max(fx) - min(fx)

    print(f"Executed in {counter} iterations")
    print(f"The minimum of the function is at {simplex[1]}.")
