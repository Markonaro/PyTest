import pandas as pd
import numpy as np
import scipy as sc
import sklearn.svm as svm
import matplotlib.pyplot as plt


def run():
    # Step 1: collect data
    file = 'data.csv'
    data = np.genfromtxt(file, delimiter=',')

    # Step 2: define parameters
    alpha = 0.0001
    theta = [0, 0]
    steps = 1000

    # Step 3: train model
    print('Starting gradient descent at  {0} = {1} + {2}x_1'.format(grad_desc(data, theta, alpha, steps), theta[0], theta[1]))
    theta = grad_desc(data, theta, alpha, steps)


def grad_desc(x, theta, alpha, steps):
    return[0, 0]


if __name__ == "__main__":
    run()
