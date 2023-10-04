import numpy as np

x0 = [1, 2]

mu = 10
epsilon = 10**(-10)
x_next = x0
diff = np.abs(x0[1]-x0[0])


# Definition of the function
def func(x):
    return (x[0]-2) ** 2 + (x[0] - 2*x[1]) ** 2

# Definition of the constraint function
def constraint(x):
    return (x[0] - x[1]**2)**2

def auxiliary(x, mu):
    return func + mu*(constraint(x))

def grad_aux(x, mu):
    return np.array[(4 + 2*mu)*x[0] - 2*(2 + mu*x[1])*x[1] - 2, 
                   -2*(1 + mu*x[1])*x[0] + 4*(1 + mu*(x[1]**2))*x[1]]

def hess_aux(x, mu):
    return np.array([[4+2*mu, 4*(1+mu*x[1])], 
                     [-2*(1+mu*x[1]), -2*mu*x[0] + 4*(1+3*mu*(x[1]**2))]])

def gradient_descent(func, x, epsilon):
    x_star = x
    norme_grad = np.linalg.norm(func(x_star), np.inf)
    while norme_grad > epsilon:
        grad = func(x_star)
        x_star = x_star - 0.001*grad
        norme_grad = np.linalg.norm(func(x_star), np.inf)
    return x_star

def newton_method(func, grad_func, hess_func, x, mu, epsilon):
    x_star = x
    norme_grad = np.linalg.norm(grad_func(x_star, mu), np.inf)
 
    while norme_grad > epsilon:
     x_star = x_star - np.inv(hess_func(x_star, mu))*grad_func(x_star, mu)
     norme_grad = np.linalg.norm(grad_func(x_star, mu))

    return x_star

if __name__ == '__main__':


    # Loop, continues until the difference between the smallest and the largest value of the evaluation of the Rosenbrock function at the simplex points is smaller than the tolerance level
    while epsilon < diff:
    
        x_next = newton_method(auxiliary, grad_aux, hess_aux, x_next, mu, epsilon)
        mu = 1/2 * mu

   # print(f"Executed in {counter} iterations")
    print(f"The constrained minimum of the function is at {x_next}.")
