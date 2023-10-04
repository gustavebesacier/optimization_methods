import numpy as np

x0 = np.array([4, 2])

mu = 10.
epsilon = 10**(-6)
x_next = x0
diff = np.abs(x0[1]-x0[0])
k=1000
i=1


# Definition of the function
def func(x):
    return (x[0]-2) ** 4 + (x[0] - 2*x[1]) ** 2

# Definition of the constraint function
def constraint(x):
    return (x[0] - x[1]**2)**2

def auxiliary(x, mu):
    return func(x) + mu*(constraint(x))

def grad_aux(x, mu):
    y = float(x[0])
    z = float(x[1])
    return np.array([4*(y-2)**3 + 2*(y - 2*z) + 2*mu*(y-z**2), 
                   -4*(y - 2*z) - 4*mu*z*(y - z**2)])

def hess_aux(x, mu):
    #print(type(x),type(mu))
    y = float(x[0])
    z = float(x[1])
    #line1 = np.array([4 + 2*mu, 4*(1 + mu*x[1])], dtype = np.float64)
    #print (line&)
    #line2 = np.array([-2*(1 + mu*x[1]), -2*mu*x[0] + 4*(1 + 3*mu*(x[1]**2))], dtype = np.float64)
    return np.array([[12*(y - 2)**2 + 2 + 2*mu, -4 - 4*mu*z],
                     [-4 - 4*mu*z, 8 - 4*mu*y + 24*mu*z**2]])
    #return np.array([line1, line2])

def gradient_descent(func, x, epsilon):
    x_star = x
    norme_grad = np.linalg.norm(func(x_star), np.inf)
    while norme_grad > epsilon:
        grad = func(x_star)
        x_star = x_star - 0.001*grad
        norme_grad = np.linalg.norm(func(x_star), np.inf)
    return x_star

def newton_method(grad_func, hess_func, x, mu, epsilon):
    x_star = x
    norme_grad = np.linalg.norm(grad_func(x_star, mu), np.inf)
    while norme_grad > epsilon:
        x_star = x_star - np.dot(np.linalg.inv(hess_func(x_star, mu)),grad_func(x_star, mu))
        norme_grad = np.linalg.norm(grad_func(x_star, mu))

    return x_star

if __name__ == '__main__':


    # Loop, continues until the difference between the smallest and the largest value of the evaluation of the Rosenbrock function at the simplex points is smaller than the tolerance level
    while i <= k:
        x_next = newton_method(grad_aux, hess_aux, x_next, mu, epsilon)
        mu = 1.1 * mu
        i = i + 1
        print(x_next)
    
    print("The constrained minimum of the function is at {x_next}.")
    print("This point verifies the constraint: x - y^2 = ", x_next[0] - x_next[1]**2)
