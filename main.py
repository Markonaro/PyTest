import random as ran
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt


def main():
    # Sample housing data
    # Size (square feet)
    # Distance to work (miles)
    # Age
    # Cost (x $1,000)

    x = np.array([[2500, 5.4, 10, 290],
                  [2100, 3.3, 85, 240],
                  [1250, 2.8, 56, 185],
                  [1725, 6.5, 23, 200],
                  [1500, 9.2, 44, 180],
                  [1800, 7.4,  2, 200]])

    plot(x[:, 0], x[:, -1], 'ro')


def plot(x, y, marker):
    plt.plot(x, y, marker)
    plt.show()

    # Until imports are used
    sc.array([0])
    print(ran.random)


if __name__ == "__main__":
    main()
