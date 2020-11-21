import matplotlib.pyplot as plt
import seaborn as sns


def compare_multiple_lines(x_array, lines, title, output_path, legend=True, xlabel=None, ylabel='Score', ylim=None, legend_position='upper right'):
    # pre: the len of the input line must have the same dimension as the x_array
    #      the input lines must have two dimensions (data, label)
    fig, ax = plt.subplots()
    for line in lines:
        data, label = line
        ax.plot(x_array, data, label=label)
    if legend:
        ax.legend(loc=legend_position, shadow=True)
    plt.title(title)
    if ylim is not None:
        plt.ylim(ylim)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight')
    plt.show()

def scatter_plot_with_hue(values, labels, cols):
    pp = sns.pairplot(values[cols], hue='cluster', size=1.8, aspect=1.8,
                      palette={"red": "#FF9999", "white": "#FFE888"},
                      plot_kws=dict(edgecolor="black", linewidth=0.5))
    fig = pp.fig
    fig.subplots_adjust(top=0.93, wspace=0.3)
    t = fig.suptitle('Wine Attributes Pairwise Plots', fontsize=14)