import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


if __name__ == '__main__':
    path = "plots"
    if not os.path.isdir(path):
        os.mkdir(path)

    plot_graph(path)
    plot_residuals(path)
    plot_2dim(path)
    plot_2dim_descriptive(path)
    plot_2dim_ols(path)
    plot_2dim_ols_with_residuals(path)
