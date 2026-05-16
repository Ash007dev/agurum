import numpy as np
from engine.ml.mmd_detector import MMDDriftDetector
mmd = MMDDriftDetector()
X = np.random.randn(2, 384)
Y = np.random.randn(2, 384)
X = X / np.linalg.norm(X, axis=1, keepdims=True)
Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
mmd2 = mmd.compute_mmd_squared(X, Y)
print("Different (2 samples):", mmd2)

# Unfloored version
n, m = len(X), len(Y)
Kxx = mmd._kernel(X, X)
Kyy = mmd._kernel(Y, Y)
Kxy = mmd._kernel(X, Y)
np.fill_diagonal(Kxx, 0.0)
np.fill_diagonal(Kyy, 0.0)
raw = Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1)) - 2.0 * Kxy.mean()
print("Raw Unfloored:", raw)

