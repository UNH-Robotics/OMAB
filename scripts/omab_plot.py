#!/usr/bin/env python3

import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import DataFrame

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
# matplotlib.rcParams['text.usetex'] = True

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
    sns.set_style("whitegrid")
    sns.set_context("talk")
    # sns.set_context("paper", rc={"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15})


def expand_dataset(data, column):
    size = len(data[column].values[0])
    return DataFrame({col: np.repeat(data[col].values, data[column].str.len())
                      for col in data.columns.difference([column])}) \
        .assign(**{column: np.concatenate(data[column].values)}) \
        .assign(Timestep=lambda x: x.index % size)


def main():
    configure_sns()
    name = "resultT"

    data = DataFrame(read_data("../results/%s.dat" % name))

    data = data.assign(experimentId=lambda df: [probability_id * 1000 + iteration for probability_id, iteration in
                                                zip(df.probabilityId, df.iteration)])
    data = data[['algorithm', 'cumSumRegrets', 'experimentId']]

    # data = data[data.algorithm != 'UCB-Value  - d0.4']
    # data = data.replace({'algorithm': {'UCB-Value  - b100 d1.0': 'UCB-Value'}})
    data = data.replace({'algorithm': {'UCB-Value  - b100 d1.0': 'UCB-Value'}})
    data = data.replace({'algorithm': {"UCB-Value l=1 a=0.4": r'UCB-Value $\mathit{lookahead} = 1$ $\alpha = 0.4$'}})
    data = data.replace({'algorithm': {"UCB-Value l=3 a=0.4": r'UCB-Value $\mathit{lookahead} = 3$ $\alpha = 0.4$'}})
    data = data.replace({'algorithm': {"Gittins-Value l=1": r'Gittins-Value $\mathit{lookahead} = 1$'}})
    data = data.replace({'algorithm': {"Gittins-Value l=3": r'Gittins-Value $\mathit{lookahead} = 3$'}})

    #
    data = expand_dataset(data, 'cumSumRegrets')

    data = data.rename(columns={"cumSumRegrets": 'Bayesian Regret'})
    print(data)
    ax = sns.tsplot(time="Timestep", value="Bayesian Regret", unit="experimentId", ci=[95], condition="algorithm",
                    data=data)
    # ax = sns.tsplot(time="Timestep", value="Bayesian Regret", unit="experimentId", ci=[95], condition="algorithm",
    #                 data=data[data.algorithm == 'Gittins'], linestyle='--')

    # ax2 = sns.tsplot(time="Timestep", value="Bayesian Regret", unit="experimentId", ci=[95], condition="algorithm",
    #                 data=data[data.algorithm != 'Gittins'], linestyle='-')
    print(ax.lines)
    print(ax.lines[0])
    sns.plt.legend(title=None, loc='upper left')

    # sns.plt.savefig('../results/%s-regret.pdf' % name)
    # sns.plt.savefig('../results/%s-regret.eps' % name)
    # sns.plt.savefig('../results/%s-regret.png' % name)

    sns.plt.show()
    # regret_plot(data)


if __name__ == "__main__":
    main()
