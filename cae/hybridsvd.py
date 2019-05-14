from __future__ import division

import random

import numpy as np
import scipy as sp
from scipy.sparse import linalg
from sksparse.cholmod import cholesky


# note: out-of-date version
# from scikits.sparse.cholmod import cholesky, CholmodError


class HybridSVD:

    def __init__(self, hybrid=True, log=True):
        """
        The module reinforces classical pure-SVD
        with additional context features for both data points and their features
        making singular-value decomposition hybrid.

        Corresponding article: https://arxiv.org/abs/1802.06398
        """
        self.hybrid = hybrid
        self.log = log
        self.components_ = None

    @staticmethod
    def get_hybrid_svd_ops(L_k, R, L_s):
        def hybrid_svd_matvec(v):
            """
            Using iterative procedure:
            u = L_k.dot(R).dot(L_s).dot(v)

            Since the below matrix multiplications cost a lot:
            A = L_k.dot(R).dot(L_s) 
            u  = A.dot(v)
            """

            u = L_k.dot(R.dot(L_s.dot(v)))
            return u

        def hybrid_svd_rmatvec(v):
            """
            (L_kRL_s)^{T}v = L_s^{T}R^{T}L_k^{T}v
            """
            u_1 = L_k.T.dot(v)
            u_2 = R.T.dot(u_1)
            u = L_s.T.dot(u_2)
            return u

        return hybrid_svd_matvec, hybrid_svd_rmatvec

    @staticmethod
    def sparse_cholesky(K):
        '''
        Compute square root of a matrix
        using sparse Choletsky decomposition
        :param K: sparse matrix
        :return L_k: L_k.dot(L_k.T) == K
        '''

        factor = cholesky(K)
        P = factor.P()
        P_inv = np.zeros(P.shape[0], dtype=np.int)
        for i in range(P.shape[0]):
            P_inv[P[i]] = i

        L = factor.L()
        L_k = np.around(
            L[P_inv[:, np.newaxis], P_inv[np.newaxis, :]], decimals=7)
        return L_k

    def compute_hybrid_pc(self):
        """
        Compute the hybrid principal components. DO NOT MAKE THE DATA ZERO MEAN!
        :return: component_[i,:] is the i-th pc
        """

        if self.log:
            print("Alpha: {}".format(self.alpha))
            print("Beta: {}".format(self.beta))
            print("NPC: {}".format(self.npc))

        # Matrix S (features matrix) square root calculation:
        self.S = sp.sparse.csc_matrix(self.S)
        # .toarray() # for sparse
        S_p = sp.sparse.identity(self.S.shape[1], format="csc") + self.alpha * self.S

        if self.alpha == 0:
            L_s = sp.sparse.identity(self.X.shape[1])
        else:
            # Note: lower triangular matrix, but not sparse
            # L_s = np.linalg.cholesky(S_p)
            L_s = HybridSVD.sparse_cholesky(S_p)

        # Matrix K (items matrix) square root calculation:
        self.K = sp.sparse.csc_matrix(self.K)
        K_p = sp.sparse.identity(self.K.shape[1], format="csc") + self.beta * self.K

        if self.beta == 0:
            L_k = sp.sparse.identity(self.X.shape[0])
        else:
            L_k = HybridSVD.sparse_cholesky(K_p)

        if self.log:
            print("type K_p: {}".format(type(K_p)))
            print("type S_p: {}".format(type(S_p)))
            print('S: {rate}% of elements are nonzero'.format(
                rate=S_p.nnz / S_p.shape[0] ** 2 * 100))
            print('K: {rate}% of elements are nonzero'.format(
                rate=K_p.nnz / K_p.shape[0] ** 2 * 100))

        # Hybrid (joint) sparse SVD calculation:
        hybrid_svd_matvec, hybrcaeid_svd_rmatvec = HybridSVD.get_hybrid_svd_ops(L_k, self.X, L_s)
        A = sp.sparse.linalg.LinearOperator(shape=self.X.shape, matvec=hybrid_svd_matvec, rmatvec=hybrcaeid_svd_rmatvec)

        # Initial vector for linalg.svd is random:
        # https://github.com/scipy/scipy/issues/7807
        random.seed(self.seed)
        np.random.seed(self.seed)
        if self.v0 is None:
            self.v0 = np.random.randn(self.X.shape[0])

        U, s, V = linalg.svds(A, k=self.npc, v0=self.v0)

        # ToDo: issue of unstability
        # ToDo: check whether abs should be taken
        if self.abs_components:
            return np.abs(V)
        return V

    def compute_pc(self):
        """
        Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
        : return: components_[i,:] is the i-th pc
        """
        from sklearn.decomposition import TruncatedSVD

        svd = TruncatedSVD(n_components=self.npc, n_iter=7,
                           random_state=self.seed)
        svd.fit(self.X)
        return svd.components_

    def fit(self, X, K, S,
            alpha=0, beta=0, npc=1, log=True, v0=None,
            abs_components=False, seed=0):
        """
        Compute hybrid principal components.
        :param X: X[i,:] is a data point
        :param K: data points context similarities
        :param S: data features context similarities
        :param alpha: matrix S influence
        :param beta: matrix K influence
        :param v0: initialization for scipy.sparse.linalg.svds
        :param npc: number of principal components to compute
        :param abs_components: todo: only for hybrid version (stability issue)
                                scipy.sparse.lingalg.svds depends on different initialization    
        :param seed: seed
        :param log: print intermediate steps
        :return: component_[i,:] is the i-th pc
        """
        self.X = X
        self.K = K
        self.S = S
        self.alpha = alpha
        self.beta = beta
        self.npc = npc
        self.v0 = v0
        self.log = log
        self.abs_components = abs_components
        self.seed = seed

        assert self.X.shape[0] == self.K.shape[0]
        assert self.X.shape[1] == self.S.shape[0]

        if self.hybrid:
            self.components_ = self.compute_hybrid_pc()
        else:
            self.components_ = self.compute_pc()
        return self.components_
