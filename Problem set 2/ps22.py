x1 = [1, 2]
x2 = [0.5, 0.8]
x3 = [2.3, 8]

# x1 = [1, 1]
# x2 = [2, 2]
# x3 = [3, 3]

alpha = 1
beta = 0.5
gamma = 2
epsilon = 10**(-10)


# Definition of the Rosenbrock function
def rosenbrock(x):
    return (1 - x[0]) ** 2 + 10 * (x[1] - x[0] ** 2) ** 2


if __name__ == '__main__':

    # List of the result of the Rosenbrock function at the 3 initial points
    fx = [rosenbrock(x1), rosenbrock(x2), rosenbrock(x3)]

    # We set the initial value of the difference between the biggest / smallest value of the Rosenbrock function evaluated at each point of the initial simplex
    diff = max(fx) - min(fx)

    # Set a counter, intialize the worse point at (0,0) and the best point where the evaluation of the Rosenbrock function is minimal
    counter = 0
    worse = [0, 0]
    best = min(fx)

    # Creation of the initial simplex
    simplex = [x1, x2, x3]

    # Loop, continues until the difference between the smallest and the largest value of the evaluation of the Rosenbrock function at the simplex points is smaller than the tolerance level
    while epsilon < diff:

        counter += 1
        res = min(fx)
        k = 0

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
