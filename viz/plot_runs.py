from viz.plot_config import *

import numpy as np


def plot_runs(xs, ys, x_label, y_label, title=None, labels=None, loglog=False, same_color=True, savefig=None):
    """ Plots a series of runs. Argsorts by label if not None.
    
    Arguments:
        xs {2D array} -- 2D array of values, shape: [num_runs, num_points_in_run]
        ys {2D array} -- 2D array of values, shape: [num_runs, num_points_in_run]
        x_label {str} -- how to label the x_axis
        y_label {str} -- how to label the y_axis
    
    Keyword Arguments:
        title {str} -- If not None, place a title on the figure (default: {None})
        labels {lia(str)} -- If not None, add labels to the legend
        loglog {bool} -- Do a log log plot (default: {False})
        same_color {bool} -- If true, plot all the lines in the same color.
        savefig {path} -- If not None, save figs to this path (includes file name) (default: {None})
    """

    plt.figure()

    if labels is not None:
        inds = np.argsort(labels)
        xs = xs[inds]
        ys = ys[inds]
        labels = [labels[i] for i in list(inds)]

    for i, (x, y) in enumerate(zip(xs, ys)):
        if same_color:
            color = all_categorical_colors[0]
        else:
            color = all_categorical_colors[(2*i) % len(all_categorical_colors)]

        plt.plot(x, y, c=color)

    if loglog:
        plt.gca().set_xscale('log')
        plt.gca().set_yscale('log')

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if title is not None:
        plt.title(title)

    if labels is not None:
        if len(labels) != len(xs):
            print('Warning - labels no the same length as the data')
        plt.legend(labels)

    plt.tight_layout()

    if savefig is not None:
        plt.savefig(savefig + '.png')
        plt.savefig(savefig + '.pdf')

    plt.close()


def plot_runs_confidence(xs, ys, x_label, y_label, title=None, labels=None, loglog=False, same_color=True, savefig=None):
    """ Plots a series of runs. Argsorts by label if not None.
    
    Arguments:
        xs {2D array} -- 3D array of values, shape: [num_configs, num_runs, num_points_in_run]
        ys {2D array} -- 3D array of values, shape: [num_configs, num_runs, num_points_in_run]
        x_label {str} -- how to label the x_axis
        y_label {str} -- how to label the y_axis
    
    Keyword Arguments:
        title {str} -- If not None, place a title on the figure (default: {None})
        labels {lia(str)} -- If not None, add labels to the legend
        loglog {bool} -- Do a log log plot (default: {False})
        same_color {bool} -- If true, plot all the lines in the same color.
        savefig {path} -- If not None, save figs to this path (includes file name) (default: {None})
    """

    plt.figure()

    if labels is not None:
        inds = np.argsort(labels)
        xs = xs[inds]
        ys = ys[inds]
        labels = [labels[i] for i in list(inds)]

    for i, (x, y) in enumerate(zip(xs, ys)):
        if same_color:
            color = all_categorical_colors[0]
        else:
            color = all_categorical_colors[i % len(all_categorical_colors)]

        y = np.array(y)
        y_mean = np.mean(y, axis=0)
        y_std = np.std(y, axis=0)
        # Must assume all y came from the same x at each time
        x = np.mean(x, axis=0)

        plt.plot(x, y_mean, c=color)
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=0.3)

    if loglog:
        plt.gca().set_xscale('log')
        plt.gca().set_yscale('log')

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if title is not None:
        plt.title(title)

    if labels is not None:
        if len(labels) != len(xs):
            print('Warning - labels no the same length as the data')
        plt.legend(labels)
    
    plt.tight_layout()

    if savefig is not None:
        plt.savefig(savefig + '.png')
        plt.savefig(savefig + '.pdf')
        
    plt.close()