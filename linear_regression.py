import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

plt.switch_backend('Qt5Agg')


def plot_graph(path):
    df = pd.read_csv("speech_quality.csv")
    print(df.head())

    x = [i for i in range(df.shape[0])]
    y = df["overall_quality"]

    mean = np.mean(y)
    print(mean)

    plt.scatter(x, y, color="#1f77b4")
    # mean fit line
    plt.axhline(mean, color="black", linestyle="--")

    plt.text(5, mean, s=f"best fit line [{mean:.1f}]", fontsize=12, verticalalignment='bottom')
    plt.title("Speech Quality")
    plt.xlabel("Wav Sample")
    plt.ylabel("Quality")
    # plt.show()
    plt.savefig(f"{path}/lr1")
    plt.close()


def plot_residuals(path):
    df = pd.read_csv("speech_quality.csv")
    print(df.head())

    x = [i for i in range(df.shape[0])]
    y = df["overall_quality"]

    mean = np.mean(y)
    residuals = [abs(mean - i) for i in y]

    plt.scatter(x, y, color="#1f77b4")
    # mean fit line
    plt.axhline(mean, color="black", linestyle="--")
    # residuals
    for x_i, y_i, r_i in zip(x, y, residuals):
        if y_i + r_i < mean:
            ymin = y_i
            ymax = mean
        else:
            ymin = mean
            ymax = y_i
        plt.vlines(x=x_i, ymin=ymin, ymax=ymax, colors='red', linestyles='--', lw=2)
        plt.text(x_i + 0.1, ((ymax - ymin) / 2) + ymin, s=f"{r_i:.1f}", horizontalalignment='left')

    plt.text(5, mean, s=f"best fit line [{mean:.1f}]"   , fontsize=12, verticalalignment='bottom')
    plt.title("Speech Quality with Residuals/Errors")
    plt.xlabel("Wav Sample")
    plt.ylabel("Quality")
    # plt.show()
    plt.savefig(f"{path}/lr2")
    plt.close()


def plot_2dim(path):
    df = pd.read_csv("speech_quality.csv")
    print(df.head())

    x = df["background_noise"].tolist()
    y = df["overall_quality"].tolist()

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    print("mean x: ", mean_x)
    print("mean y: ", mean_y)

    plt.scatter(x, y)
    plt.title("Speech Quality")
    plt.xlabel("background_noise")
    plt.ylabel("overall_quality")
    plt.savefig(f"{path}/lr3")
    plt.close()


def plot_2dim_descriptive(path):
    df = pd.read_csv("speech_quality.csv")
    print(df.head())

    x = df["background_noise"].tolist()
    y = df["overall_quality"].tolist()

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    print("mean x: ", mean_x)
    print("mean y: ", mean_y)

    plt.scatter(x, y)
    plt.scatter(mean_x, mean_y, s=100, c='black', marker="x")
    plt.text(mean_x + 2, mean_y - 1, f"[{mean_x}, {mean_y}]")
    plt.title("Speech Quality")
    plt.xlabel("background_noise")
    plt.ylabel("overall_quality")
    plt.savefig(f"{path}/lr4")
    plt.close()


def plot_2dim_ols(path):
    df = pd.read_csv("speech_quality.csv")
    print(df.head())

    x = df["background_noise"].tolist()
    y = df["overall_quality"].tolist()

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    print("mean x: ", mean_x)
    print("mean y: ", mean_y)

    numerator = 0
    denominator = 0
    for x_i, y_i in zip(x, y):
        numerator += (x_i - mean_x) * (y_i - mean_y)
        denominator += (x_i - mean_x) ** 2
    b1 = numerator / denominator
    b0 = mean_y - b1 * mean_x
    f_x = [x_i * b1 + b0 for x_i in x]
    fx_string = f"f(x) = {b1:.1f}x + {b0:.1f}"

    plt.scatter(x, y)
    plt.plot(x, f_x, c="green")
    plt.scatter(mean_x, mean_y, s=100, c='black', marker="x")

    plt.text(20, 50, fx_string, c="green")
    plt.text(mean_x + 2, mean_y - 1, f"[{mean_x}, {mean_y}]")
    plt.title("Speech Quality")
    plt.xlabel("background_noise")
    plt.ylabel("overall_quality")
    plt.savefig(f"{path}/lr5")
    plt.close()


def plot_2dim_ols_with_residuals(path):
    df = pd.read_csv("speech_quality.csv")
    print(df.head())

    x = df["background_noise"].tolist()
    y = df["overall_quality"].tolist()

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    print("mean x: ", mean_x)
    print("mean y: ", mean_y)

    numerator = 0
    denominator = 0
    for x_i, y_i in zip(x, y):
        numerator += (x_i - mean_x) * (y_i - mean_y)
        denominator += (x_i - mean_x) ** 2
    b1 = numerator / denominator
    b0 = mean_y - b1 * mean_x
    f_x = [x_i * b1 + b0 for x_i in x]
    fx_string = f"f(x) = {b1:.1f}x + {b0:.1f}"

    plt.scatter(x, y)
    plt.plot(x, f_x, c="green")
    plt.scatter(mean_x, mean_y, s=100, c='black', marker="x")

    residuals = [r - y_i for y_i, r in zip(y, f_x)]
    print(y)
    print(f_x)
    residuals = [round(i, 2) for i in residuals]
    print(residuals)
    squared_residuals = [round(i ** 2, 2) for i in residuals]
    print(squared_residuals)
    print(sum(squared_residuals))
    for x_i, y_i, f_x_i, r_i in zip(x, y, f_x, residuals):
        if r_i < 0:
            ymin = y_i
            ymax = f_x_i
        else:
            ymin = f_x_i
            ymax = y_i
        plt.vlines(x=x_i, ymin=ymin, ymax=ymax, colors='red', linestyles='--', lw=2)
        plt.text(x_i + 1, y_i, s=f"{abs(r_i):.1f}", horizontalalignment='left')

    plt.text(20, 50, fx_string, c="green")
    plt.text(mean_x + 2, mean_y - 1, f"[{mean_x}, {mean_y}]")
    plt.title("Speech Quality")
    plt.xlabel("background_noise")
    plt.ylabel("overall_quality")
    plt.savefig(f"{path}/lr6")
    plt.close()


def plot_3dim_mlr(path):
    df = pd.read_csv("speech_quality.csv")
    print(df.head())

    x = df["background_noise"].tolist()
    y = df["delay"].tolist()
    z = df["overall_quality"].tolist()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z)

    A = np.vstack([x, y, np.ones_like(x)]).T
    plane_coef, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

    # Create a meshgrid for the plane
    x_plane, y_plane = np.meshgrid(np.unique(x), np.unique(y))
    z_plane = plane_coef[0] * x_plane + plane_coef[1] * y_plane + plane_coef[2]

    # Add the regression plane
    ax.plot_surface(x_plane, y_plane, z_plane, alpha=0.5)

    # Add labels and title
    ax.set_xlabel('Noise')
    ax.set_ylabel('Delay')
    ax.set_zlabel('Quality Rating')
    plt.title('Multiple Linear Regression')

    plt.savefig(f"{path}/lr7")
    plt.close()


def plot_mlr(path):
    df = pd.read_csv("speech_quality.csv")
    x1 = np.array(df["background_noise"])
    x2 = np.array(df["delay"])
    y = df["overall_quality"].tolist()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x1, x2, y)

    # 2d
    A_2d = np.vstack([x1, np.ones(len(x1))]).T
    solution_2d, _, _, _ = np.linalg.lstsq(A_2d, y, rcond=None)
    ols_line = np.linalg.lstsq(A_2d, y, rcond=None)[0]
    line = ols_line[0] * x1 + ols_line[1]
    residuals = [r - line_i for line_i, r in zip(line, y)]
    ax.plot(x1, line, zs=9, zdir='y', label="OLS line", color="green")

    for x_i, r_i, y_i, line_i in zip(x1, residuals, y, line):
        if r_i < 0:
            ymin = y_i
            ymax = line_i
        else:
            ymin = line_i
            ymax = y_i
        ax.plot([x_i, x_i], [ymin, ymax], color='red', zs=9, zdir='y')

    # 3d
    A = np.vstack([x1, x2, np.ones(len(x1))])
    solution_3d, residuals, _, _ = np.linalg.lstsq(A.T, y, rcond=None)
    print(solution_3d)
    print(residuals)
    xxx = np.linalg.lstsq(A.T, y, rcond=None)[0]
    xx = np.dot(A.T, xxx)
    individual_residuals = y - xx
    print(individual_residuals)
    x1_plane, x2_plane = np.meshgrid(np.unique(x1), np.unique(x2))
    z_plane = solution_3d[0] * x1_plane + solution_3d[1] * x2_plane + solution_3d[2]
    ax.plot_surface(x1_plane, x2_plane, z_plane, alpha=0.5)

    for x1_i, x2_i, y_i, r_i in zip(x1, x2, y, individual_residuals):
        if r_i < 0:
            ymin = y_i
            ymax = y_i - r_i
        else:
            ymin = y_i - r_i
            ymax = y_i
        ax.plot(x1_i, x2_i, [ymin, ymax], color='red')

    # plot stuff
    ax.set_xlabel('Noise')
    ax.set_ylabel('Delay')
    ax.set_zlabel('Quality Rating')
    plt.title('Multiple Linear Regression')
    plt.savefig(f"{path}/lr8")
    plt.close()


def plot_non_linear_lr(path):
    df = pd.read_csv("speech_quality.csv")
    print(df.head())

    x = np.array(df["background_noise"])
    y = df["non_linear_overall_quality"].tolist()

    # linear
    A = np.vstack([x, np.ones(len(x))]).T
    solution_2d, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    ols_line = np.linalg.lstsq(A, y, rcond=None)[0]
    line = ols_line[0] * x + ols_line[1]
    residuals = [r - line_i for line_i, r in zip(line, y)]
    # plt.plot(x, line, label="OLS line", color="green", alpha=0.5)

    # non-linear
    coeff = [25, -1.5, 0.05, -0.0005]
    c, cov = curve_fit(non_linear_formula, x, y)
    print(coeff)
    print(c)
    X = np.linspace(0, 100, 50)
    pred = [non_linear_formula(xi, *c) for xi in X]

    plt.scatter(x, y)
    plt.plot(X, pred, "r")

    # plot stuff
    plt.xlabel('Noise')
    plt.ylabel('Quality Rating')
    plt.title('Non-linear Regression')
    # plt.savefig(f"{path}/lr9")
    plt.show()
    plt.close()


def non_linear_formula(xi, b0, b1, b2, b3):
    return b0 + b1 * xi + b2 * (xi ** 2) + b3 * (xi ** 3)


if __name__ == '__main__':
    path = "plots"
    if not os.path.isdir(path):
        os.mkdir(path)

    """plot_graph(path)
    plot_residuals(path)
    plot_2dim(path)
    plot_2dim_descriptive(path)
    plot_2dim_ols(path)
    plot_2dim_ols_with_residuals(path)
    plot_3dim_mlr(path)
    plot_mlr(path)"""
    plot_non_linear_lr(path)
