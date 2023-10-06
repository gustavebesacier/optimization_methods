import numpy as np
from time import perf_counter
from scipy.linalg import inv, solve, det

def func(x):
    """Objective function"""
    return (x[0]-2) ** 4 + (x[0] - 2*x[1]) ** 2

def constraint(x):
    """Constraint function"""
    return x[0] - x[1]**2

def auxiliary(x, mu):
    """Auxiliary function"""
    return func(x) + mu*(constraint(x))**2

def grad_aux(x, mu):
    """Gradient of the auxiliary function"""
    y = x[0]
    z = x[1]
    return np.array([4*(y-2)**3 + 2*(y - 2*z) + 2*mu*(y-z**2), 
                   -4*(y - 2*z) - 4*mu*z*(y - z**2)])

def hess_aux(x, mu):
    """Hessian of the auxiliary function"""
    y = x[0]
    z = x[1]
    return np.array([[12*(y - 2)**2 + 2 + 2*mu, -4 - 4*mu*z],
                     [-4 - 4*mu*z, 8 - 4*mu*y + 24*mu*z**2]])

def gradient_descent(func, x, epsilon):
    x_star = x
    norme_grad = np.linalg.norm(func(x_star), np.inf)
    while norme_grad > epsilon:
        grad = func(x_star)
        x_star = x_star - 0.001*grad
        norme_grad = np.linalg.norm(func(x_star), np.inf)
    return x_star

def newton_method(grad_func, hess_func, x, mu, epsilon):
    """Newton method to find a minimum"""
    x_star = x
    norme_grad = np.linalg.norm(grad_func(x_star, mu), np.inf)
    k = 0
    while norme_grad > epsilon and k < 501:
        k += 1
        start = perf_counter()
        # x_star = x_star - np.dot(np.linalg.inv(hess_func(x_star, mu)),grad_func(x_star, mu))
        d = solve(hess_func(x_star, mu), grad_func(x_star, mu), assume_a="sym")
        print(det(hess_func(x_star, mu)))
        x_star -= d
        norme_grad = np.linalg.norm(grad_func(x_star, mu), np.inf)
        stop = perf_counter()
        # print(f"\t iteration {k}: {1000*(stop-start)} ms")

    return x_star, k

def penalty_opt(x0, tolerance, mu, factor_mu, epsilon, grad_func=grad_aux, hess_func=hess_aux):
    k = 0 
    x_k = x0

    print(abs(constraint(x_k)))
    while abs(x_k[0]-x_k[1]**2) > tolerance:
        k += 1
        print(f"{k} th call to Newton", end="")
        start = perf_counter()
        x_k, k_newton = newton_method(grad_func, hess_func, x_k, mu, epsilon)
        stop = perf_counter()
        print(f" took {k_newton} iterations in {1000*(stop-start)} ms (or {stop-start} s)")
        #hess_det = det(hess_func(x_k, mu))
        #print(f"hess det: {hess_det}, mu: {mu}, abs(constraint): {abs(constraint(x_k))}, x_k = {x_k}")
        mu = factor_mu * mu

    return k, x_k

if __name__ == '__main__':

    point_0 = np.array([2.2, 1.5])

    mu0 = 10.0
    epsilon0 = 1e-10
    tolerance0 = 1e-10

    #x_next = point_0
    #diff = np.abs(point_0[1]-point_0[0])
    k=100
    i=1

    print(penalty_opt(x0=point_0, tolerance=tolerance0, mu=mu0, factor_mu=2, epsilon=epsilon0))

    # while i <= k:
    #    x_next = newton_method(grad_aux, hess_aux, x_next, mu, epsilon)
     #   mu = 1.1 * mu
    #    i = i + 1
    #    print(x_next)
    
    # print("The constrained minimum of the function is at {x_next}.")
    # print("This point verifies the constraint: x - y^2 = ", x_next[0] - x_next[1]**2)
