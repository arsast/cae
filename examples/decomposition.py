import os

import numpy as np

from cae.hybridsvd import HybridSVD

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32,OMP_NUM_THREADS=4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

X = np.random.rand(100, 200)
K = np.identity(X.shape[0])
S = np.identity(X.shape[1])

S[1, 6] = 1
S[6, 1] = 1

hsvd = HybridSVD(hybrid=True)
V_hybrid = hsvd.fit(X, K, S, alpha=0, beta=0,
                    abs_components=True)

hsvd = HybridSVD(hybrid=False)
V_pure = hsvd.fit(X, K, S)

# note: results comparison between pure and hybrid SVD
assert type(V_pure) == type(V_hybrid)
assert V_pure.shape == V_hybrid.shape
np.testing.assert_allclose(V_pure, V_hybrid)

S[1, 6] = 1
S[6, 1] = 1

hsvd = HybridSVD(hybrid=True, log=False)
V_hybrid = hsvd.fit(X, K, S, alpha=0.3, beta=0.3, log=False, seed=15)

print(V_hybrid)
