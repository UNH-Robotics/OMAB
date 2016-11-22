#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas import DataFrame

__author__ = 'Bence Cserna'


def construct_data_frame(data):
    flat_data = [flatten(experiment) for experiment in data]
    return DataFrame(flat_data)


def regret_box_plot(experiments):
    boxplot = sns.boxplot(x="algorithm",
                          y="regret",
                          data=experiments,
                          showmeans=True)
    plt.show(boxplot)


def regret_plot(data):
    regret_series = data[['algorithm', 'regrets']].groupby('algorithm').apply(list_average, 'regrets')
    values = []
    for row in regret_series.values:
        values.append([value for value in row])

    max_len = 0
    for row in values:
        max_len = max(len(row), max_len)

    for row in values:
        if len(row) < max_len:
            row += [float('NaN')] * (max_len - len(row))

    frame = DataFrame(np.asarray(values).T, columns=list(regret_series.index))
    print(frame)
    # plt.figure()

    frame.plot()
    plt.show()


def read_data(file_name):
    with open(file_name) as file:
        content = json.load(file)
    return content


def list_average(group, key):
    return [sum(col) / float(len(col)) for col in zip(*group[key])]


def configure_sns():
    sns.set_style("white")


def main():
    configure_sns()
    data = DataFrame(read_data("../../experiment-results/result_vi_ts_ucb_100.dat"))
    data2 = DataFrame(read_data("../../experiment-results/2-arm-SS2-wTS.data"))



# regret_box_plot(data)
    # data.set_index('algorithm', inplace=True)
    data = data.append(data2, ignore_index=True)
    regret_plot(data)


if __name__ == "__main__":
    main()
