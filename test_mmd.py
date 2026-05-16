import numpy as np
from engine.ml.mmd_detector import MMDDriftDetector
mmd = MMDDriftDetector()
X = np.random.randn(3, 384)
Y = np.random.randn(3, 384)
X = X / np.linalg.norm(X, axis=1, keepdims=True)
Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
print("Different:", mmd.compute_mmd_squared(X, Y))
print("Identical:", mmd.compute_mmd_squared(X, X))
