import numpy as np


def bounding_box(X):
    """ Calculates the bounding box of the data.

    Args:
        X: The data

    Returns:
        Two arrays: mins, maxs. Mins contains the minimum values in
        each dimension of the data. Maxs contains the maximum values
        in each dimension of the data.
    """
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    return mins, maxs


def make_uniform_distribution(n, mins, maxs, random_state=None):
    """ Creates a uniform distribution in a bounding box.

    Args:
        n (integer): The number of datapoints
        mins: np.array of minimums for each dimension of the bounding box
        maxs: np.array of maximums for each dimension of the bounding box

    Returns:
        A n x len(mins) np.array containing the new datapoints
    """
    if random_state is not None:
        np.random.seed(random_state)
    X = np.empty(shape=(n, len(mins)))
    for i in range(n):
        for j in range(len(mins)):
            try:
                X[i][j] = np.random.uniform(mins[j], maxs[j])
            except IndexError:
                raise IndexError(('Maxs does not have the same'
                                  ' dimensions as mins'))
    return X
