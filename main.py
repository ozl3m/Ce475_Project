import numpy as np
import csv
import pandas as pd
import sklearn.ensemble as sen
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score


def mse(y, y_pred):
    return np.mean((y - y_pred)**2)


if __name__ == '__main__':

    with open("data.csv") as f:
        csv_list = list(csv.reader(f))


    x1 = np.array([])
    x2 = np.array([])
    x3 = np.array([])
    x4 = np.array([])
    x5 = np.array([])
    x6 = np.array([])
    Y = np.array([])

 # Extracting data into lists, creating X and y:
    for row in csv_list[1:]:
      #  x1 = np.append(x1, int(row[1]))
        x2 = np.append(x2, int(row[2]))
        x3 = np.append(x3, int(row[3]))
        x4 = np.append(x4, int(row[4]))
        x5 = np.append(x5, int(row[5]))
        x6 = np.append(x6, int(row[6]))
        Y = np.append(Y, int(row[7]))

 # Forming the input(X) and the output(y)
    X = np.column_stack((x1, x2, x3, x4, x5, x6))
    y = Y
