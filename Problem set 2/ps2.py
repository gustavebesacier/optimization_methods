x1 = [1, 1]
x2 = [2, 2]
x3 = [3, 3]

alpha = 1
beta = 0.5
gamma = 2
epsilon = 1e-6


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 10 * (x[1] - x[0] ** 2) ** 2


if __name__ == '__main__':

    print(rosenbrock(x2))

    fx = [rosenbrock(x1), rosenbrock(x2), rosenbrock(x3)]

    diff = max(fx) - min(fx)
    counter = 0

    # while diff > epsilon:
    while counter < 10:

        counter += 1
        res = rosenbrock(x1)
        simplex = [x1, x2, x3]
        k=0
        worse = [0, 0]

        for i in range(len(simplex)):
            print("Boucle for i, i=", i)
            if fx[i] >= res:
                print("fx[i]",fx[i])
                res = fx[i]
                worse = simplex[i]
                print("Worse ", worse)
                k = i
                print("Dans la condition if")
        simplex.remove(simplex[k])
        print("Simplex remove", simplex)

        c = [
            (simplex[0][0] + simplex[1][0])/2,
            (simplex[0][1] + simplex[1][1]) / 2
        ]

        x_try = [(1 + alpha) * c[0] - alpha * c[0], (1 + alpha) * c[1] - alpha * c[1]]

        if rosenbrock(x_try) > rosenbrock(worse):
            x_new = [(1 - beta) * c[0] + beta * worse[0], (1 - beta) * c[1] + beta * worse[1]]
        else:
            x_new = [(1 + gamma) * x_try[0] - gamma * x_try[0], (1 + gamma) * x_try[1] - gamma * x_try[1]]

        simplex.append(x_new)

        diff = max(fx) - min(fx)

        print(counter)
        print(simplex)

    print(counter)
    print(simplex)