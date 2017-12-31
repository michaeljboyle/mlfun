# Gap Statistic Method
""" The gap statistic is a measure to determine the optimal number of clusters
for a dataset. The primary method is gap_statistc, which calculates the
gap statistic for each number of clusters k, where the gap statistic is
given by the formula:
    Gap stat(k) = Gap(k) - (Gap(k+1) - s(k+1))
This value is used to determine the optimal # of clusters when
Gap stat(k) first becomes positive.
"""

# Imports
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


def __wk(X, labels):
    """ Computes the sum of squared distances from each point in X
    to the mean of its corresponding cluster.

    Args:
        X: the data, a np.array
        labels: the assigned cluster label for each data point, a np.array

    Returns:
        The sum of squared distances from each point in X to the mean
            of its corresponding cluster.
    """
    unique_labels = np.unique(labels)
    n_samples_per_label = np.bincount(labels, minlength=len(unique_labels))
    intra_clust_dists = np.zeros(len(unique_labels))
    for curr_label in range(len(unique_labels)):
        mask = labels == curr_label
        current_distances = pairwise_distances(X[mask]) ** 2
        n_samples_curr_lab = n_samples_per_label[curr_label]
        if n_samples_curr_lab != 0:
            intra_clust_dists[curr_label] = \
                np.sum(current_distances) / (2 * n_samples_curr_lab)
    return np.sum(intra_clust_dists)


def __bounding_box(X):
    """ Calculates the bounding box of the data.

    Args:
        X: The data

    Returns:
        Two lists: mins, maxs. Mins contains the minimum values in
        each dimension of the data. Maxs contains the maximum values
        in each dimension of the data.
    """
    n_dims = len(X.shape)
    mins = []
    maxs = []
    for dim in range(n_dims):
        mins.append(min(X, key=lambda a: a[dim])[dim])
        maxs.append(max(X, key=lambda a: a[dim])[dim])
    return mins, maxs


def gap_statistic(X, label_list, clusterers, return_graph_stats=False):
    """ Calculates gap statistic for each cluster.

    Args:
        X: the data that has been clustered.
        label_list: A list of np.arrays containing the assigned labels
            as assigned by the clusterer. Each np.array in the list
            corresponds to a different number of clusters used in the
            clustering, ordered in ascending order.
        clusterers: A list of clusterer instances used to create
            the labels in label_list. Should correspond 1:1 with
            each array of labels in label_list
        return_graph_stats: Boolean which determines whether additional
            stats will be returned, useful for graphing purposes

    Returns:
        A np.array of the gap statistic for each number of clusters k with
        dimensions 2 x len(k), where the first array are the values of k
        and the second array is the corresponding gap statistic.
        Gap stat(k) = Gap(k) - (Gap(k+1) - s(k+1))
        This value is used to determine the optimal # of clusters when
        Gap stat(k) first becomes positive.
        If return_graph_stats is True, also returns:
            array of k's, array of sk's, array of logWk's, array of AvgLogWkb's
    """
    ks = np.array([len(np.unique(labels)) for labels in label_list])
    mins, maxs = __bounding_box(X)

    logWks = np.zeros(len(ks))  # Holds the log(Wk) for each k
    avgLogWkbs = np.zeros(len(ks))  # Holds average Wkb for each k over all B
    sks = np.zeros(len(ks))  # holds sk for each k

    # Create B reference datasets
    B = 10
    Xbs = []  # holds each reference dataset
    for i in range(B):
        Xb = []
        for n in range(len(X)):
            distribs = []
            for j in range(len(mins)):
                distribs.append(np.random.uniform(mins[j], maxs[j]))
            Xb.append(distribs)
        Xb = np.array(Xb)
        Xbs.append(Xb)
    # Cluster B
    for indk in range(len(ks)):
        logWk = np.log(__wk(X, label_list[indk]))
        logWks[indk] = logWk
        clusterer = clusterers[indk]
        logWkbs = np.zeros(B)
        for i, Xb in enumerate(Xbs):
            Xb_labels = clusterer.fit_predict(Xb)
            logWkbs[i] = np.log(__wk(Xb, Xb_labels))
        avgLogWkb = np.sum(logWkbs) / B
        avgLogWkbs[indk] = avgLogWkb
        sks[indk] = np.sqrt(sum((logWkbs - avgLogWkb) ** 2) / B)
    sks = sks * np.sqrt(1 + 1 / B)  # make final conversion of SD to sk term
    gaps = avgLogWkbs - logWks

    # Ignore last index of shifted because it is meaningless
    gap_stats = (gaps - (np.roll(gaps, -1) - np.roll(sks, -1)))[:-1]
    if not return_graph_stats:
        return np.stack((ks[: -1], gap_stats))
    else:
        return np.stack((ks[: -1], gap_stats)), ks, sks, logWks, avgLogWkbs


def graph_gap_statistic(X, label_list, clusterers):
    """ Graphs the gap statistic along with supporting data.

    Args:
        X: the data that has been clustered.
        label_list: A list of np.arrays containing the assigned labels
            as assigned by the clusterer. Each np.array in the list
            corresponds to a different number of clusters used in the
            clustering, ordered in ascending order.
        clusterers: A list of clusterer instances used to create
            the labels in label_list. Should correspond 1:1 with
            each array of labels in label_list

    Returns:
        None
    """
    gap_stats, ks, sks, logWks, avgLogWkbs = gap_statistic(
        X, label_list, clusterers, return_graph_stats=True)
    gap_stats_values = gap_stats[1]

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    # First plot Wk vs Ks
    ax1 = axs[0][0]
    ax1.grid(c='white')
    ax1.plot(ks, np.exp(logWks), 'g-o')
    ax1.set(xlabel='Number of clusters K',
            ylabel='Wk (sum intra-cluster distance^2 to cluster mean)')

    # Plot Wk and Wkbs vs Ks
    ax2 = axs[0][1]
    ax2.grid(c='white')
    ax2.plot(ks, logWks, 'b-o', ks, avgLogWkbs, 'r-o')
    ax2.set(xlabel='Number of clusters K', ylabel='Log Wk and Log WkB')

    # Plot gap vs Ks
    ax3 = axs[1][0]
    ax3.grid(c='white')
    ax3.plot(ks, avgLogWkbs - logWks, 'g-o')
    ax3.set(xlabel='Number of clusters K', ylabel='Gap')

    # Plot gap diff b/w k and k+1 accounting for sd
    ax4 = axs[1][1]
    ax4.bar(ks[: -1], gap_stats_values, 0.8, alpha=1.0, color='b')
    ax4.set_xticks(ks)
    ax4.set(xlabel='Number of clusters K',
            ylabel='Gap(k) - (Gap(k+1) - s(k+1))')

    plt.show()
