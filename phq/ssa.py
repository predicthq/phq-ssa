import logging

import numpy as np
from scipy import linalg

log = logging.getLogger(__name__)

MINIMUM_TIMESERIES_DATA_LENGTH = 100


def _get_contributions(singular_values):
    """
    Calculate the relative contribution of each of the positive singular values
    :param singular_values: numpy array
    :return: Dataframe with positive Contribution values
    """
    lambdas = np.power(singular_values, 2)
    contributions = lambdas / lambdas.sum()
    return contributions[contributions > 0]


def _diagonal_averaging(trajectory_matrix):
    """
    Performs anti-diagonal averaging from given trajectory matrix
    :param trajectory_matrix: numpy 2D array
    :return: Pandas DataFrame object containing the reconstructed series
    """
    row_count, col_count = trajectory_matrix.shape

    if row_count > col_count:
        # Transpose the matrix
        trajectory_matrix = trajectory_matrix.T
        row_count, col_count = col_count, row_count

    result = []
    # Diagonal Averaging
    for k in range(1 - col_count, row_count):
        mask = np.eye(col_count, k=k, dtype='bool')[::-1][:row_count, :]
        ma = np.ma.masked_array(trajectory_matrix.A, mask=(1 - mask))
        result += [ma.sum() / mask.sum()]

    return np.array(result)


def embed(time_series_data, embedding_dimension=None):
    """
    Embed the time series with embedding_dimension window size.
    :param time_series_data: Numpy array
    :param embedding_dimension: int
    :return: trajectory matrix (Numpy matrix)
    """

    if not isinstance(time_series_data, np.ndarray):
        raise TypeError('time_series_data should be an instance of Numpy array')

    timeseries_data_count = time_series_data.shape[0]

    if timeseries_data_count < MINIMUM_TIMESERIES_DATA_LENGTH:
        raise ValueError(f'time_series_data length ({timeseries_data_count}) is smaller than the required minimum length of {MINIMUM_TIMESERIES_DATA_LENGTH}')

    if not np.isfinite(time_series_data).all():
        raise TypeError(f'time_series_data contains one or more infinite values')
    if np.sum(np.abs(time_series_data)) == 0:
        raise ValueError('time_series_data should not be all zeros')

    default_embedding_dimension = timeseries_data_count // 2
    if embedding_dimension and embedding_dimension > default_embedding_dimension:
        log.warning('Provided embedding dimension %(embedding_dimension)d is greater than default embeddiing_dimension %(default_embedding_dimension)d',
                    {
                        'embedding_dimension': embedding_dimension,
                        'default_embedding_dimension': default_embedding_dimension
                    })
    elif not embedding_dimension:
        embedding_dimension = default_embedding_dimension
    log.debug('Embedding dimension: %(dimension)s', {'dimension': embedding_dimension})

    k = timeseries_data_count - embedding_dimension + 1
    trajectory_matrix = np.matrix(linalg.hankel(time_series_data, np.zeros(embedding_dimension))).T[:, :k]
    return trajectory_matrix


def reconstruction(trajectory_matrix, contribution_proportion):
    """
    Reconstruction based on the trajectory matrix/matrices and contributionprop
    :param trajectory_matrix: trajectory_matrix
    :param contribution_proportion: the percentage of energy for reconstruction
    """
    u, s, _ = linalg.svd(trajectory_matrix * trajectory_matrix.T)
    u = np.matrix(u)
    s = np.sqrt(s)

    ssa_s_contributions = _get_contributions(s)
    sum_ssa = [sum(ssa_s_contributions[0:i + 1]) for i in range(ssa_s_contributions.shape[0])]

    nsig = np.argmin([abs(i - contribution_proportion) for i in sum_ssa]) + 1

    xs = np.zeros(trajectory_matrix.shape)
    for i in range(nsig):
        vi = trajectory_matrix.T * (u[:, i] / s[i])
        yi = s[i] * u[:, i]
        xs = xs + (yi * np.matrix(vi).T)
    r = len(ssa_s_contributions)
    log.debug('Rank of trajectory: %(rank)d', {'rank': r})
    log.debug('Rank of reconstructed of trajectory matrix: %(nsig)s', {'nsig': nsig})

    characteristic = round((s[:r] ** 2).sum() / (s ** 2).sum(), 4)
    log.debug('Characteristic of projection: %(characteristics)f', {'characteristics': characteristic})
    return _diagonal_averaging(xs), nsig
