import numpy as np
import pandas as pd
from numpy import matrix as m
from pandas import DataFrame as df
from scipy import linalg


class SSA(object):
    '''Singular Spectrum Analysis object'''

    def __init__(self, time_series):

        self.ts = pd.DataFrame(time_series)
        self.ts_name = self.ts.columns.tolist()[0]
        if self.ts_name == 0:
            self.ts_name = 'ts'
        self.ts_v = self.ts.values
        self.ts_N = self.ts.shape[0]
        self.freq = self.ts.index.inferred_freq

    @staticmethod
    def _printer(name, *args):
        '''Helper function to print messages neatly'''
        print('-' * 40)
        print(name + ':')
        for msg in args:
            print(msg)

    @staticmethod
    def _dot(x, y):
        '''Alternative formulation of dot product to allow missing values in arrays/matrices'''
        pass

    @staticmethod
    def get_contributions(X=None, s=None):
        '''Calculate the relative contribution of each of the singular values'''
        lambdas = np.power(s, 2)
        frob_norm = np.linalg.norm(X)
        ret = df(lambdas / (frob_norm ** 2), columns=['Contribution'])
        ret['Contribution'] = ret.Contribution.round(4)
        return ret[ret.Contribution > 0]

    @staticmethod
    def diagonal_averaging(hankel_matrix):
        '''Performs anti-diagonal averaging from given hankel matrix
        Returns: Pandas DataFrame object containing the reconstructed series'''
        mat = m(hankel_matrix)
        L, K = mat.shape
        L_star, K_star = min(L, K), max(L, K)
        new = np.zeros((L, K))
        if L > K:
            mat = mat.T
        ret = []

        # Diagonal Averaging
        for k in range(1 - K_star, L_star):
            mask = np.eye(K_star, k=k, dtype='bool')[::-1][:L_star, :]
            mask_n = sum(sum(mask))
            ma = np.ma.masked_array(mat.A, mask=1 - mask)
            ret += [ma.sum() / mask_n]

        return df(ret).rename(columns={0: 'Reconstruction'})

    def embed(self, embedding_dimension=None, suspected_frequency=None, verbose=False, return_df=False):
        '''Embed the time series with embedding_dimension window size.
        Optional: suspected_frequency changes embedding_dimension such that it is divisible by suspected frequency'''
        if not embedding_dimension:
            self.embedding_dimension = self.ts_N // 2
        else:
            self.embedding_dimension = embedding_dimension
        if suspected_frequency:
            self.suspected_frequency = suspected_frequency
            self.embedding_dimension = (self.embedding_dimension // self.suspected_frequency) * self.suspected_frequency

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

        if verbose:
            msg1 = 'Embedding dimension\t:  {}\nTrajectory dimensions\t: {}'
            msg2 = 'Complete dimension\t: {}\nMissing dimension     \t: {}'
            msg1 = msg1.format(self.embedding_dimension, self.trajectory_dimentions)
            msg2 = msg2.format(self.complete_dimensions, self.missing_dimensions)
            self._printer('EMBEDDING SUMMARY', msg1, msg2)

        if return_df:
            return self.X_df

    def decompose(self, verbose=False):
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
        self.s_contributions = self.get_contributions(X, self.s)
        self.r = len(self.s_contributions[self.s_contributions > 0])
        self.r_characteristic = round((self.s[:self.r] ** 2).sum() / (self.s ** 2).sum(), 4)
        self.orthonormal_base = {i: self.U[:, i] for i in range(self.r)}

        if verbose:
            msg1 = 'Rank of trajectory\t\t: {}\nDimension of projection space\t: {}'
            msg1 = msg1.format(self.d, self.r)
            msg2 = 'Characteristic of projection\t: {}'.format(self.r_characteristic)
            self._printer('DECOMPOSITION SUMMARY', msg1, msg2)

    @classmethod
    def reconstruction(cls, *hankel):
        '''Reconstruction of the hankel matrix/matrices passed to *hankel'''
        hankel_mat = None
        for han in hankel:
            if isinstance(hankel_mat, m):
                hankel_mat = hankel_mat + han
            else:
                hankel_mat = han.copy()
        hankel_full = cls.diagonal_averaging(hankel_mat)
        return hankel_full
