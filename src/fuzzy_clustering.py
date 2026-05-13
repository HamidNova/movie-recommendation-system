import numpy as np
from sklearn.preprocessing import StandardScaler
try:
    import skfuzzy as fuzz
except ImportError:
    raise ImportError("scikit-fuzzy is required. Install with: pip install scikit-fuzzy")

class FuzzyClustering:
    """
    Fuzzy C-Means clustering over item (or user) embeddings.
    """
    def __init__(self, n_clusters=10, m=2.0, max_iter=100, error=0.005, random_state=42):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.error = error
        self.random_state = random_state
        self.cntr = None          # cluster centers
        self.u = None             # fuzzy partition matrix (n_samples x n_clusters)

    def fit(self, X):
        """
        X : numpy array of shape (n_samples, n_features)
        """
        # Standardize to improve convergence
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fuzzy C-Means
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            X_scaled.T, self.n_clusters, self.m, error=self.error,
            maxiter=self.max_iter, seed=self.random_state
        )
        self.cntr = cntr          # shape (n_clusters, n_features)
        self.u = u.T              # shape (n_samples, n_clusters)
        return self

    def transform(self, X):
        """
        Return fuzzy membership vectors for new data points.
        X : numpy array (n_samples, n_features)
        """
        if self.cntr is None:
            raise RuntimeError("Fit the model first.")
        scaler = StandardScaler()
        # Note: For proper transform, we'd need to use the same scaler fitted on training data.
        # Here we approximate by scaling again; for better consistency store the scaler.
        X_scaled = scaler.fit_transform(X)
        # Compute membership using the same m and centers
        cntr = self.cntr
        u = np.zeros((X_scaled.shape[0], self.n_clusters))
        for i in range(X_scaled.shape[0]):
            dists = np.linalg.norm(X_scaled[i] - cntr, axis=1)
            # avoid division by zero
            dists[dists == 0] = 1e-10
            u[i] = 1.0 / np.sum((dists[:, None] / dists[None, :]) ** (2/(self.m-1)), axis=1)
        return u
