from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from ...utils.data_utils import bounding_box, make_uniform_distribution
import numpy as np


def hopkins(X, random_state=None):
    """ Calculates the Hopkins statistic for the data distribution.
    Values > 0.5 suggest that the data is not uniformly distributed
    and could be clusterable.

    Args:
        X: A np.array of data
        random_state: int or None, for reproducibility
    Returns:
        A float 0 < x < 1 indicating whether data is potentially clusterable.
    """

    # Get subset of X (5%)
    _, Xn = train_test_split(
        X, test_size=0.05, random_state=random_state)
    n = Xn.shape[0]

    # Create random uniform distribution with n points in same space as X
    mins, maxs = bounding_box(X)
    R = make_uniform_distribution(n, mins, maxs, random_state=random_state)

    nbrs = NearestNeighbors(n_neighbors=2).fit(X)

    # Get nearest neighbors in X for points in Xn
    Ws = nbrs.kneighbors(Xn)[0][:, 1]

    # Get nearest neighbors in X for points in R
    Us = nbrs.kneighbors(R, n_neighbors=1)[0][:, 0]

    try:
        sumUs = np.sum(Us)
        H = sumUs / (sumUs + np.sum(Ws))
    except ZeroDivisionError:
        H = 0

    return H




    # d = X.shape[1]
    # n = len(X)  # rows
    # m = int(0.1 * n)  # heuristic from article [1]
    # nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)

    # rand_X = sample(range(0, n, 1), m)

    # ujd = []
    # wjd = []
    # for j in range(0, m):
    #     u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
    #     ujd.append(u_dist[0][1])
    #     w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
    #     wjd.append(w_dist[0][1])

    # H = sum(ujd) / (sum(ujd) + sum(wjd))
    # if isnan(H):
    #     print ujd, wjd
    #     H = 0

    # return H