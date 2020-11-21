import numpy as np
from prettytable import PrettyTable
from src.auxiliary.evaluation_methods import run_evaluation_metric
from src.auxiliary.file_methods import save_csv
import sklearn.metrics as metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
import itertools


def show_charts(config, output_path, values, labels, output_labels, visualize, dataframe, verbose):
    for chart in config:
        eval(chart['name'])(chart, output_path, values, labels, output_labels, visualize, dataframe, verbose)


def class_frequencies(config, output_path, values, labels, output_labels, visualize, dataframe, verbose):
    unique1, counts1 = np.unique(labels, return_counts=True)
    #counts1, unique1 = zip(*sorted(zip(counts1, unique1), reverse=True)) -> show sorted
    unique2, counts2 = np.unique(output_labels, return_counts=True)

    rows = [['class', 'original_samples_distribution', 'predicted_samples_distribution']]
    x = PrettyTable()
    x.field_names = ['class', 'original_samples_distribution', 'predicted_samples_distribution']
    for index, _class in enumerate(unique1):
        number_samples1 = counts1[index]
        number_samples2 = counts2[np.where(unique2 == _class)[0]][0]
        rows.append([_class, number_samples1, number_samples2])
        x.add_row([_class, number_samples1, number_samples2])
    if output_path is not None:
        save_csv(output_path + '/samples_distribution', rows)
    print(x)

def class_frequencies_separate(config, output_path, values, labels, output_labels, visualize, dataframe, verbose):
    unique, counts = np.unique(output_labels, return_counts=True)
    rows = [['class', 'predicted_samples_distribution']]
    x = PrettyTable()
    x.field_names = ['class', 'predicted_samples_distribution']
    for index, _class in enumerate(unique):
        number_samples = counts[index]
        rows.append([_class, number_samples])
        x.add_row([_class, number_samples])
    if output_path is not None:
        save_csv(output_path + '/samples_distribution', rows)
    print(x)


def show_metrics_table(config, output_path, values, labels, output_labels, visualize, dataframe, verbose):
    rows = [[''] + config['metrics']]
    x = PrettyTable()
    x.field_names = [''] + config['metrics']
    scores = []
    for metric in config['metrics']:
        scores.append(run_evaluation_metric(metric, values, labels, output_labels))
    rows.append(['result'] + scores)
    x.add_row(['result'] + scores)
    if output_path is not None:
        save_csv(output_path + '/table_scores', rows)
    print(x)


def show_classification_report(config, output_path, values, labels, output_labels, visualize, dataframe, verbose):
    print(metrics.classification_report(labels, output_labels))
    classification_report = metrics.classification_report(labels, output_labels, output_dict=True)
    classification_report = pd.DataFrame(classification_report).transpose()
    classification_report.to_csv(output_path + '/classification_report.csv', index=False)


def show_confusion_matrix(config, output_path, values, labels, output_labels, visualize, dataframe, verbose):
    print('Confusion matrix')
    print(metrics.confusion_matrix(labels, output_labels))
    confusion_matrix = metrics.confusion_matrix(labels, output_labels)
    confusion_matrix = pd.DataFrame(confusion_matrix).transpose()
    confusion_matrix.to_csv(output_path + '/confusion_matrix.csv', index=False)


def show_feature_histograms(config, output_path, values, labels, output_labels, visualize, dataframe, verbose):
    dataframe.hist(bins=config['bins'], color='steelblue', edgecolor='black', linewidth=1.0, xlabelsize=8, ylabelsize=8, grid=False)
    if output_path is not None:
        plt.savefig(output_path + '/feature_histograms', bbox_inches='tight')
    plt.show()


def show_correlation_among_variables(config, output_path, values, labels, output_labels, visualize, dataframe, verbose):
    f, ax = plt.subplots(figsize=(config['figsize'][0], config['figsize'][1]))
    corr = dataframe.corr()
    hm = sns.heatmap(round(corr, 2), annot=True, ax=ax, cmap="coolwarm", fmt='.2f',linewidths=.05)
    f.subplots_adjust(top=0.93)
    t = f.suptitle(config['title'], fontsize=14)
    if output_path is not None:
        plt.savefig(output_path + '/correlation_heatmap', bbox_inches='tight')
    plt.show()


def show_parallel_coordinates(config, output_path, values, labels, output_labels, visualize, dataframe, verbose):
    f, ax = plt.subplots(figsize=(config['figsize'][0], config['figsize'][1]))
    plt.title(config['title'] + ' original labels')
    pc = parallel_coordinates(dataframe, dataframe.columns[-1], color=('#FFE888', '#FF9999'))
    if output_path is not None:
        plt.savefig(output_path + '/parallel_coordinates_labels', bbox_inches='tight')
    plt.show()
    aux = dataframe.copy()
    aux[dataframe.columns[-1]] = output_labels
    f, ax = plt.subplots(figsize=(config['figsize'][0], config['figsize'][1]))
    plt.title(config['title'] + ' predicted labels')
    pc = parallel_coordinates(aux, dataframe.columns[-1])
    if output_path is not None:
        plt.savefig(output_path + '/parallel_coordinates_predicted_labels', bbox_inches='tight')
    plt.show()


def show_pair_wise_scatter_plot(config, output_path, values, labels, output_labels, visualize, dataframe, verbose):

    if config['original']:
        pp = sns.pairplot(dataframe, hue=dataframe.columns[-1], height=1.8, aspect=1.8, plot_kws=dict(edgecolor="black", linewidth=0.5))
        fig = pp.fig
        fig.subplots_adjust(top=0.93, wspace=0.3)
        t = fig.suptitle(config['title']  + ' for the original labels', fontsize=14)
        if output_path is not None:
            plt.savefig(output_path + '/pair_wise_scatter_plot_labels', bbox_inches='tight')
        plt.show()

    if config['predicted']:
        aux = dataframe.copy()
        aux[dataframe.columns[-1]] = output_labels
        pp = sns.pairplot(aux, hue=aux.columns[-1], height=1.8, aspect=1.8,
                          plot_kws=dict(edgecolor="black", linewidth=0.5))
        fig = pp.fig
        fig.subplots_adjust(top=0.93, wspace=0.3)
        t = fig.suptitle(config['title'] + ' for the predicted labels', fontsize=14)
        if output_path is not None:
            plt.savefig(output_path + '/pair_wise_scatter_plot_predicted_labels', bbox_inches='tight')
        plt.show()


def show_clusters_2d(config, output_path, values, labels, output_labels, visualize, dataframe, verbose):
    number_clusters = len(np.unique(labels))

    figsize = [10, 20] if 'figsize' not in config else config['figsize']

    clusters = []
    for i in range(number_clusters):
        cluster_indices = [x for x in range(labels.shape[0]) if labels[x] == i]
        clusters.append(np.take(values, cluster_indices, axis=0))

    f, ax = plt.subplots(number_clusters + 1, 1, figsize=(figsize[0], figsize[1]))
    ax[0].set_title(config['title'])
    colors = itertools.cycle(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"])
    for index, cluster in enumerate(clusters):
        color = next(colors)
        ax[0].scatter(cluster[:,0], cluster[:,1], color=color, label='Class ' + str(index))
        ax[index+1].scatter(cluster[:, 0], cluster[:, 1], color=color)
        ax[index+1].set_ylabel('Dimension 0')
        ax[index+1].set_xlabel('Dimension 1')
    ax[0].legend()
    ax[0].set_ylabel('Dimension 0')
    ax[0].set_xlabel('Dimension 1')
    if output_path is not None:
        plt.savefig(output_path + '/2d_cluster_plots', bbox_inches='tight')
    plt.show()

