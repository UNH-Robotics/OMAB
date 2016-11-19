#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

__author__ = 'Bence Cserna'


def construct_data_frame(data):
    flat_data = [flatten(experiment) for experiment in data]
    return DataFrame(flat_data)


def regret_box_plot(experiments):
    sns.set_style("white")

    boxplot = sns.boxplot(x="algorithm",
                          y="regret",
                          data=experiments,
                          showmeans=True)
    plt.show(boxplot)


def test(experiments):
    experiments.plot()


def read_data(file_name):
    with open(file_name) as file:
        content = json.load(file)
    return content


def main():
    data = DataFrame(read_data("../results/result4.dat"))
    # regret_box_plot(data)
    # data.set_index('algorithm', inplace=True)

    print(data[['algorithm', 'reward']].groupby('algorithm').)


if __name__ == "__main__":
    main()
