from numpy import *
import random as ran
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib as style


# Calculate the total difference between each data point
# and the current hypothesis function.
def err(theta, data):
    # Define some values
    j = 0
    m = len(data)

    # Isolate each data point's x and y values.
    # Add the difference between each x and y to the total difference variable, j.
    for i in range(m):
        x = data[i, 0]
        y = data[i, -1]
        j += (theta[1]*x + theta[0] - y)**2
    return j / 2*m


def step_gradient(theta, data, a):
    # Gradient Descent
    grads = [0, 0]
    m = len(data)

    for i in range(m):
        x = data[i, 0]
        y = data[i, -1]

        grads[0] += (theta[1]*x + theta[0] - y) / m
        grads[1] += (x * (theta[1]*x + theta[0] - y)) / m

    # Adjust each theta proportionate to the learning rate and gradient
    theta[0] = theta[0] - (a * grads[0])
    theta[1] = theta[1] - (a * grads[1])

    return theta


def grad_desc(data, theta, a, it):
    # Step toward the local minimum for every designated iteration
    for i in range(it):
        theta = step_gradient(theta, data, a)
    return theta


def linear_train(th, d, a, i):
    # Step 3: Train model
    print("Initial hypothesis:\nh(x) = {1:.2f}x + {0:.2f}\nerror = {2:.2f}\n".format(
        th[0], th[1], err(th, d)))
    e1 = err(th, d)

    print("Running...\n")

    # Determine the optimal theta values
    theta = grad_desc(d, th, a, i)

    print("Final hypothesis:\nh(x) = {1:.2f}x + {0:.2f}\nerror = {2:.2f}\niterations = {3}\n".format(
        th[0], th[1], err(th, d), i))
    e2 = err(th, d)

    print('Error reduction: {0:.2f}%'.format(100 * (1 - e2 / e1)))


def run():
    # Step 1: Collect data
    data = genfromtxt('data.csv', delimiter=',')

    # Step 2: Define hyper parameters
    alpha = 0.00000001
    iterations = 10000

    theta = [1]  # Bias unit
    for t in range(len(data[0]) - 1):
        theta.append(ran.random())

    linear_train(theta, data, alpha, iterations)

if __name__ == "__main__":
    run()
