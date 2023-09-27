import copy

# x1 = [1, 2]
# x2 = [0.5, 0.8]
# x3 = [2.3, 8]

# x1 = [1, 1]
# x2 = [2, 2]
# x3 = [3, 3]

# x1 = [1983, 200]
# x2 = [9873, 98656]
# x3 = [2.3, 8]

x1 = [1982, 201]
x2 = [9872, 98655]
x3 = [2.3, 8]

alpha = 1
beta = 0.5
gamma = 2
epsilon = 1e-12


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 10 * (x[1] - x[0] ** 2) ** 2


if __name__ == '__main__':

    fx = [rosenbrock(x1), rosenbrock(x2), rosenbrock(x3)]

    diff = max(fx) - min(fx)
    counter = 0
    simplex = [x1, x2, x3]
    worse = [0, 0]
    best = min(fx)

    while epsilon < diff:

        counter += 1
        res = min(fx)
        k = 0

        for i in range(len(simplex)):

            if fx[i] >= res:

                res = fx[i]
                worse = simplex[i]
                k = i

        simplex.remove(worse)

        c = [(simplex[0][0] + simplex[1][0]) / 2, (simplex[0][1] + simplex[1][1]) / 2]

        x_try = [(1 + alpha) * c[0] - alpha * worse[0], (1 + alpha) * c[1] - alpha * worse[1]]

        if rosenbrock(x_try) > rosenbrock(worse):
            x_new = [(1 - beta) * c[0] + beta * worse[0], (1 - beta) * c[1] + beta * worse[1]]
        elif rosenbrock(x_try) < best:
            x_new = [(1 + gamma) * x_try[0] - gamma * c[0], (1 + gamma) * x_try[1] - gamma * c[1]]
        else:
            x_new = copy.deepcopy(x_try)

        simplex.append(x_new)
        fx = [rosenbrock(simplex[0]), rosenbrock(simplex[1]), rosenbrock(simplex[2])]
        best = min(fx)
        diff = max(fx) - min(fx)

    print(f"Executed in {counter} iterations")
    print(f"The of the function is {simplex}")
