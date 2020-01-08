from functools import reduce
import logging

import numpy as np
import pandas as pd
from scipy import linalg

log = logging.getLogger(__name__)


class SSA:
    """
    Singular Spectrum Analysis object
    """
    def __init__(self, time_series):
        self._timeseries_data = pd.DataFrame(time_series)

    @staticmethod
    def get_contributions(singular_values):
        """
        Calculate the relative contribution of each of the positive singular values
        :param singular_values: numpy array
        :return: Dataframe with positive Contribution values
        """
        lambdas = np.power(singular_values, 2)
        contributions = lambdas / lambdas.sum()                
        return contributions[contributions> 0]

    @staticmethod
    def diagonal_averaging(trajectory_matrix):
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

        ret = []
        # Diagonal Averaging
        for k in range(1 - col_count, row_count):
            mask = np.eye(col_count, k=k, dtype='bool')[::-1][:row_count, :]
            ma = np.ma.masked_array(trajectory_matrix.A, mask=(1 - mask))
            ret += [ma.sum() / mask.sum()]

        return pd.DataFrame(ret, columns=['Reconstruction'])

    def embed(self, embedding_dimension=None):
        """
        Embed the time series with embedding_dimension window size.
        :param embedding_dimension:
        :return:
        """
        timeseries_data_row_count = self._timeseries_data.shape[0]

        if not embedding_dimension:
            embedding_dimension = timeseries_data_row_count // 2

        log.debug('Embedding dimension: %(dimension)s', {'dimension': embedding_dimension})

        k = timeseries_data_row_count - embedding_dimension + 1
        x = np.matrix(linalg.hankel(self._timeseries_data, np.zeros(embedding_dimension))).T[:, :k]
        return x

    def reconstruction(self, x, contributionprop):
        """
        Reconstruction based on the trajectory matrix/matrices and contributionprop 
        :param x: trajectory_matrix
        :param contributionprop: the percentage of energy for reconstruction
        """
        u, s, v = linalg.svd(x * x.T)
        u = np.matrix(u)
        s = np.sqrt(s)
        v = np.matrix(v)
        d = np.linalg.matrix_rank(x)
        log.debug('Dimension of projection space: %(dimension)d', {'dimension': d})
        
        ssa_s_contributions = self.get_contributions(s)
        sum_ssa = []
        for i in range(ssa_s_contributions.shape[0]):
            a =sum(ssa_s_contributions[0:i])
            sum_ssa.append(a) 
        t1 = [i - contributionprop for i in sum_ssa]
        t2 = np.abs(t1)
        nsig = np.argmin(t2)
        xs = np.zeros(x.shape)
        for i in range(nsig):
            vi = x.T * (u[:, i] / s[i])
            yi = s[i] * u[:, i]
            xs = xs + yi * np.matrix(vi).T
        r = len(ssa_s_contributions)
        log.debug('Rank of trajectory: %(rank)d', {'rank': r})

        characteristic = round((s[:r] ** 2).sum() / (s ** 2).sum(), 4)
        log.debug('Characteristic of projection: %(characteristics)f', {'characteristics': characteristic})
        return self.diagonal_averaging(xs), nsig