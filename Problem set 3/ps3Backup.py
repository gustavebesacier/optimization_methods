import numpy as np


def func(x):
    """Objective function"""
    return (x[0] - 2) ** 4 + (x[0] - 2 * x[1]) ** 2


def constraint(x):
    """Constraint function"""
    return x[0] - x[1] ** 2


def auxiliary(x, mu):
    """Auxiliary function"""
    return func(x) + mu * (constraint(x)) ** 2


def grad_aux(x, mu):
    """Gradient of the auxiliary function"""
    """Returns a column matrix"""
    y = float(x[0])
    z = float(x[1])
    # return np.array([4 * (y - 2) ** 3 + 2 * (y - 2 * z) + 2 * mu * (y - z ** 2),
    #                  -4 * (y - 2 * z) - 4 * mu * z * (y - z ** 2)])
    return np.array([[4 * (y - 2) ** 3 + 2 * (y - 2 * z) + 2 * mu * (y - z ** 2)],
                         [-4 * (y - 2 * z) - 4 * mu * z * (y - z ** 2)]])


def hess_aux(x, mu):
    """Hessian matrix of the auxiliary function"""
    y = float(x[0])
    z = float(x[1])
    # line1 = np.array([4 + 2*mu, 4*(1 + mu*x[1])], dtype = np.float64)
    # print (line&)
    # line2 = np.array([-2*(1 + mu*x[1]), -2*mu*x[0] + 4*(1 + 3*mu*(x[1]**2))], dtype = np.float64)
    return np.array([[12 * (y - 2) ** 2 + 2 + 2 * mu, -4 - 4 * mu * z],
                     [-4 - 4 * mu * z, 8 - 4 * mu * y + 24 * mu * z ** 2]])
    # return np.array([line1, line2])


def gradient_descent(func, x, epsilon):
    x_star = x
    norme_grad = np.linalg.norm(func(x_star), np.inf)
    while norme_grad > epsilon:
        grad = func(x_star)
        x_star = x_star - 0.001 * grad
        norme_grad = np.linalg.norm(func(x_star), np.inf)
    return x_star


def newton_method(grad_func, hess_func, x, mu, epsilon):
    x_star = x
    x = grad_func(x_star, mu)
    # norme_grad = np.linalg.norm(grad_func(x_star, mu), np.inf)
    norme_grad = np.linalg.norm(x, np.inf)

    while norme_grad > epsilon:
        x_star = x_star - np.dot(np.linalg.inv(hess_func(x_star, mu)), grad_func(x_star, mu))
        norme_grad = np.linalg.norm(grad_func(x_star, mu))

    return x_star


def penalty_opt(x0, tolerance, mu, factor_mu, epsilon, grad_func=grad_aux, hess_func=hess_aux):
    k = 0
    x_k = x0

    while abs(constraint(x_k)) > tolerance:
        k += 1
        x_k = newton_method(grad_func, hess_func, x_k, mu, epsilon)
        mu = factor_mu * mu

    return x_k, k


if __name__ == '__main__':

    x0 = np.array([1, 1])

    print(func(x0))
    print(grad_aux(x0,1))
    print(hess_aux(x0,1))

    mu = 5
    epsilon = 10 ** (-6)
    x_next = x0
    diff = np.abs(x0[1] - x0[0])
    k = 100
    i = 1
    counter =0

    norme_grad = np.linalg.norm(grad_aux(x0, mu), np.inf)
    print(norme_grad)

    print(np.dot(np.linalg.inv(hess_aux(x0, mu)), grad_aux(x0, mu)))

    print(newton_method(grad_aux(x0,5), hess_aux(x0,5),x0,5,epsilon))
    # Loop, continues until the difference between the smallest and the largest value of the evaluation of the Rosenbrock function at the simplex points is smaller than the tolerance level
    # while i <= k:
    #     x_next = newton_method(grad_aux, hess_aux, x_next, mu, epsilon)
    #     counter += 1
    #     mu = 1.1 * mu
    #     i = i + 1
    #     print(x_next)
    #
    # print(f"The constrained minimum of the function is at {x_next}.")
    # print("This point verifies the constraint: x - y^2 = ", x_next[0] - x_next[1] ** 2)
    # print("counter ", counter)
    # print("Mu ", mu)