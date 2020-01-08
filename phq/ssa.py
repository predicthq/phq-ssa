from functools import reduce
import logging

import numpy as np
import pandas as pd
from pandas import DataFrame as df
from scipy import linalg

log = logging.getLogger(__name__)


class SSA(object):
    """
    Singular Spectrum Analysis object
    """
    def __init__(self, time_series):

        self.ts = pd.DataFrame(time_series)
        self.ts_name = self.ts.columns.tolist()[0]
        if self.ts_name == 0:
            self.ts_name = 'ts'
        self.ts_v = self.ts.values
        self.ts_N = self.ts.shape[0]
        self.freq = self.ts.index.inferred_freq

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

        if not embedding_dimension:
            embedding_dimension = self.ts_N // 2

        if suspected_frequency:
            embedding_dimension = (embedding_dimension // suspected_frequency) * suspected_frequency

        self.K = self.ts_N - self.embedding_dimension + 1
        self.X = m(linalg.hankel(self.ts, np.zeros(self.embedding_dimension))).T[:, :self.K]
        self.X_df = df(self.X)
        self.X_complete = self.X_df.dropna(axis=1)
        self.X_com = m(self.X_complete.values)
        self.X_missing = self.X_df.drop(self.X_complete.columns, axis=1)
        self.X_miss = m(self.X_missing.values)
        self.trajectory_dimentions = self.X_df.shape
        self.complete_dimensions = self.X_complete.shape
        self.missing_dimensions = self.X_missing.shape
        self.no_missing = self.missing_dimensions[1] == 0

        log.debug('Embedding dimension: %(dimension)s', {'dimension': str(self.embedding_dimension) })
        log.debug('Trajectory dimensions: %(dimension)s', {'dimension': str(self.trajectory_dimentions)})
        log.debug('Complete dimensions: %(dimension)s', {'dimension': str(self.complete_dimensions)})
        log.debug('Missing dimensions: %(dimension)s', {'dimension': str(self.missing_dimensions)})

        return self.X_df

    def decompose(self):
        '''Perform the Singular Value Decomposition and identify the rank of the embedding subspace
        Characteristic of projection: the proportion of variance captured in the subspace'''
        X = self.X_com
        self.S = X * X.T
        self.U, self.s, self.V = linalg.svd(self.S)
        self.U, self.s, self.V = m(self.U), np.sqrt(self.s), m(self.V)
        self.d = np.linalg.matrix_rank(X)
        Vs, Xs, Ys, Zs = {}, {}, {}, {}
        for i in range(self.d):
            Zs[i] = self.s[i] * self.V[:, i]
            Vs[i] = X.T * (self.U[:, i] / self.s[i])
            Ys[i] = self.s[i] * self.U[:, i]
            Xs[i] = Ys[i] * (m(Vs[i]).T)
        self.Vs, self.Xs = Vs, Xs
        self.s_contributions = self.get_contributions(self.s)
        self.r = len(self.s_contributions[self.s_contributions > 0])
        self.r_characteristic = round((self.s[:self.r] ** 2).sum() / (self.s ** 2).sum(), 4)
        self.orthonormal_base = {i: self.U[:, i] for i in range(self.r)}

        log.debug('Rank of trajectory: %(rank)d', {'rank': self.r})
        log.debug('Dimension of projection space: %(dimension)d', {'dimension': self.d})
        log.debug('Characteristic of projection: %(characteristics)f', {'characteristics': self.r_characteristic})

    def reconstruction(self, *trajectory_matrices):
        """
        Reconstruction of the trajectory matrix/matrices
        :param trajectory_matrices: (List of) Numpy 2D array
        """
        return self.diagonal_averaging(reduce(np.add, trajectory_matrices))
