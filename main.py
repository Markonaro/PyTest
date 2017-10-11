import pandas as pd
import numpy as np
import scipy as sc
import sklearn.svm as svm
import matplotlib.pyplot as plt


def main():
    file = 'data.txt'
    X, y = get_data(file)
    # plt.plot(X[:, 0], y, 'ro')
    # plt.show()


def get_data(filename):
    data = pd.read_csv(filename)
    return np.array(data.ix[:, :-1].values),\
        np.array(data.ix[:, -1].values)


if __name__ == "__main__":
    main()
