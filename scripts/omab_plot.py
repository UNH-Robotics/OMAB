#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import DataFrame
from scipy import stats

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
    regret_series = data[['algorithm', 'cumSumRegrets']].groupby('algorithm').apply(list_average,
                                                                                    'cumSumRegrets')

    print(regret_series)
    values = []
    for row in regret_series.values:
        values.append([value for value in row])

    max_len = 0
    for row in values:
        max_len = max(len(row), max_len)

    # Fill missing values with NaN
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


def list_average_cum_sum(group, key):
    averages = [sum(col) / float(len(col)) for col in zip(*group[key])]
    cum_sums = [averages[x] * (1 + x) for x in range(len(averages))]

    return cum_sums


def configure_sns():
    sns.set_style("white")


def probability_index(probabilities):
    print(probabilities)
    multiplier = 1
    index = 0
    for p in probabilities:
        index += round(p * 100) * multiplier
        multiplier *= 10
    return index


def expand_dataset(data, column):
    size = len(data[column].values[0])
    return DataFrame({col: np.repeat(data[col].values, data[column].str.len())
                      for col in data.columns.difference([column])})\
        .assign(**{column: np.concatenate(data[column].values)})\
        .assign(step=lambda x: x.index % size)


def main():
    configure_sns()
    data = DataFrame(read_data("../results/resultT.dat"))

    data = data.assign(pindex=lambda df: [probability_index(probabilities) for probabilities in df.probabilities])
    regrets_ = data[['algorithm', 'cumSumRegrets', 'pindex']]
    expanded = expand_dataset(regrets_, 'cumSumRegrets')
    print(expanded)
    ax = sns.tsplot(time="step", value="cumSumRegrets", unit="pindex", condition="algorithm", data=expanded)
    sns.plt.show()

    # regret_plot(data)


if __name__ == "__main__":
    main()
