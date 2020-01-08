from functools import reduce
import logging

import numpy as np
import pandas as pd
from pandas import DataFrame as df
from scipy import linalg

log = logging.getLogger(__name__)


class SSA:
    """
    Singular Spectrum Analysis object
    """
    def __init__(self, time_series):
        self._timeseries_data = pd.DataFrame(time_series)

    @staticmethod
    def _dot(x, y):
        """
        Alternative formulation of dot product to allow missing values in arrays/matrices
        """
        pass

    @staticmethod
    def get_contributions(singular_values):
        """
        Calculate the relative contribution of each of the singular values
        :param singular_values: numpy array
        :return: Dataframe with positive Contribution values
        """
        lambdas = np.power(singular_values, 2)
        contributions_df = df((lambdas / lambdas.sum()).round(4), columns=['Contribution'])
        # Only returns positive contribution values
        return contributions_df[contributions_df.Contribution > 0]

    @staticmethod
    def diagonal_averaging(trajectory_matrix):
        """
        Performs anti-diagonal averaging from given trajectory matrix
        :param trajectory_matrix: numpy 2D array
        :return: Pandas DataFrame object containing the reconstructed series
        """
        matrix = np.matrix(trajectory_matrix)
        row_count, col_count = matrix.shape

        if row_count > col_count:
            # Transpose the matrix
            matrix = np.matrix.T
            row_count, col_count = col_count, row_count

        ret = []
        # Diagonal Averaging
        for k in range(1 - col_count, row_count):
            mask = np.eye(col_count, k=k, dtype='bool')[::-1][:row_count, :]
            ma = np.ma.masked_array(matrix.A, mask=(1 - mask))
            ret += [ma.sum() / mask.sum()]

        return df(ret, columns=['Reconstruction'])

    def embed(self, embedding_dimension=None, suspected_frequency=None):
        """
        Embed the time series with embedding_dimension window size.
        Optional: suspected_frequency changes embedding_dimension such that it is divisible by suspected frequency'''
        :param embedding_dimension:
        :param suspected_frequency:
        :return:
        """
        timeseries_data_row_count = self._timeseries_data.shape[0]

        if not embedding_dimension:
            embedding_dimension = timeseries_data_row_count // 2

        if suspected_frequency:
            embedding_dimension = (embedding_dimension // suspected_frequency) * suspected_frequency

        log.debug('Embedding dimension: %(dimension)s', {'dimension': embedding_dimension})

        k = timeseries_data_row_count - embedding_dimension + 1
        x = np.matrix(linalg.hankel(self._timeseries_data, np.zeros(embedding_dimension))).T[:, :k]
        x_df = df(x)
        log.debug('Trajectory dimensions: %(dimension)s', {'dimension': x_df.shape})

        x_complete = x_df.dropna(axis=1).values
        log.debug('Complete dimensions: %(dimension)s', {'dimension': x_complete.shape})

        x_missing = x_df.drop(x_complete.columns, axis=1)
        log.debug('Missing dimensions: %(dimension)s', {'dimension': x_missing.shape})

        return {
            'x': x_df,
            'x_complete': x_complete,
            'x_missing': x_missing
        }

    def decompose(self, x):
        """
        Perform the Singular Value Decomposition and identify the rank of the embedding subspace
        Characteristic of projection: the proportion of variance captured in the subspace
        """
        u, s, v = linalg.svd(x * x.T)
        u = np.matrix(u)
        s = np.sqrt(s)
        v = np.matrix(v)
        d = np.linalg.matrix_rank(x)
        log.debug('Dimension of projection space: %(dimension)d', {'dimension': d})

        vs, xs, ys, zs = {}, {}, {}, {}
        for i in range(d):
            zs[i] = s[i] * v[:, i]
            vs[i] = x.T * (u[:, i] / s[i])
            ys[i] = s[i] * u[:, i]
            xs[i] = ys[i] * np.matrix(vs[i]).T

        contributions = self.get_contributions(s)
        r = len(contributions[contributions > 0])
        log.debug('Rank of trajectory: %(rank)d', {'rank': r})

        characteristic = round((s[:r] ** 2).sum() / (s ** 2).sum(), 4)
        log.debug('Characteristic of projection: %(characteristics)f', {'characteristics': characteristic})

        return vs, xs, ys, zs

    def reconstruction(self, *trajectory_matrices):
        """
        Reconstruction of the trajectory matrix/matrices
        :param trajectory_matrices: (List of) Numpy 2D array
        """
        return self.diagonal_averaging(reduce(np.add, trajectory_matrices))
