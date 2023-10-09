import numpy as np
from time import perf_counter
from scipy.linalg import inv, solve, det
import matplotlib.pyplot as plt
import math

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
    return np.array([4*(y-2)**3 + 2*(y - 2*z) + 2*mu*(y-z**2), -4*(y - 2*z) - 4*mu*z*(y - z**2)])

def hess_aux(x, mu):
    """Hessian of the auxiliary function"""
    y = x[0]
    z = x[1]
    #return np.array([[12*(y - 2)**2 + 2 + 2*mu, -4 - 4*mu*z], [-4 - 4*mu*z, 8 - 4*mu*y + 24*mu*z**2]])
    return np.array([[12*(y - 2)**2 + 2 + 2*mu, -4 - 4*mu*z], [-4 - 4*mu*z, 8 - 4*mu*y + 12*mu*z**2]])

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
        #print(det(hess_func(x_star, mu)))
        x_star -= d
        norme_grad = np.linalg.norm(grad_func(x_star, mu), np.inf)
        stop = perf_counter()
        # print(f"\t iteration {k}: {1000*(stop-start)} ms")

    return x_star, k

def penalty_opt(x0, tolerance, mu, factor_mu, epsilon, grad_func=grad_aux, hess_func=hess_aux):
    k = 0 
    x_k = x0

    #print(abs(constraint(x_k)))
    while abs(x_k[0]-x_k[1]**2) > tolerance:
        k += 1
        #print(f"{k}th call to Newton", end="\n")
        start = perf_counter()
        x_k, k_newton = newton_method(grad_func, hess_func, x_k, mu, epsilon)
        stop = perf_counter()
        #print(f"Optimized in {k_newton} iterations in {1000*(stop-start)}ms ({stop-start}s)")
        #hess_det = det(hess_func(x_k, mu))
        #print(f"hess det: {hess_det}, mu: {mu}, abs(constraint): {abs(constraint(x_k))}, x_k = {x_k}")
        mu = factor_mu * mu

    return k, x_k, mu

if __name__ == '__main__':

    point_0 = np.array([2.2, 1.5])

    mu0 = 10.0
    epsilon0 = 1e-10
    tolerance0 = 1e-10

    mu_vec = [10, 11, 12, 15, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 10000000]
    x_1 = list()
    x_2 = list()
    t = list()

    for i in mu_vec:
        point_0 = np.array([2.2, 1.5])
        epsilon0 = 1e-10
        tolerance0 = 1e-10
        start = perf_counter()
        iteration, coor, mu = penalty_opt(x0=point_0, tolerance=tolerance0, mu=i, factor_mu=2, epsilon=epsilon0)
        end = perf_counter()
        print(f"Mininum at {coor[0], coor[1]} in {iteration} iterations, mu = {i}/{mu} (intial/final). Time : {end-start}s")
        x_1.append(float(coor[0]))
        x_2.append(float(coor[1]))
        t.append(float(end-start))
    print(x_1, "\n", x_2)

    m = [math.log(i) for i in mu_vec]
    plt.plot(m, x_1, label="Value of x_1")
    plt.plot(m, x_2, label="Value of x_2")
    plt.xlabel("log(mu)")
    plt.ylabel("min of the function")
    plt.twinx()
    plt.plot(m, t, c="r", label="Duration")
    plt.title("Values of the minimum as a function of mu (in logs).")
    plt.annotate("Blue: x_1; Orange: x_2; Red: duration in s (right axis)", (0.5, -0.13), xycoords='axes fraction', fontsize=8, ha='center')
    plt.show()

    # iteration, coor = penalty_opt(x0=point_0, tolerance=tolerance0, mu=mu0, factor_mu=2, epsilon=epsilon0)
    # print(f"Mininum at {coor[0],coor[1]}, found in {iteration} iterations.")
