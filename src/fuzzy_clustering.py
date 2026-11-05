import os

import joblib
import numpy as np
import skfuzzy as fuzz

from sklearn.preprocessing import StandardScaler


class FuzzyClustering:

    def __init__(
        self,
        n_clusters=5,
        m=1.5,
        max_iter=300,
        error=1e-5,
        random_state=42
    ):
        if not isinstance(n_clusters, (int, np.integer)):
            raise TypeError(
                "n_clusters must be an integer."
            )

        if n_clusters < 2:
            raise ValueError(
                "n_clusters must be at least 2."
            )

        if m <= 1.0:
            raise ValueError(
                "m must be greater than 1.0 for fuzzy c-means."
            )

        if not isinstance(max_iter, (int, np.integer)):
            raise TypeError(
                "max_iter must be an integer."
            )

        if max_iter <= 0:
            raise ValueError(
                "max_iter must be greater than 0."
            )

        if error <= 0:
            raise ValueError(
                "error must be greater than 0."
            )

        self.n_clusters = int(n_clusters)
        self.m = float(m)
        self.max_iter = int(max_iter)
        self.error = float(error)
        self.random_state = random_state

        self.scaler = StandardScaler()

        self.centers = None
        self.membership = None
        self.fpc = None

    def _validate_embeddings(self, embeddings):

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D array with shape "
                "(n_samples, n_features)."
            )

        if embeddings.shape[0] == 0:
            raise ValueError(
                "Embeddings cannot be empty."
            )

        if embeddings.shape[1] == 0:
            raise ValueError(
                "Embeddings must contain at least one feature."
            )

        if not np.isfinite(embeddings).all():
            raise ValueError(
                "Embeddings contain NaN or infinite values."
            )

        return embeddings

    def _prepare_embeddings(
        self,
        embeddings,
        fit=False
    ):

        embeddings = self._validate_embeddings(
            embeddings
        )

        if fit:
            X_scaled = self.scaler.fit_transform(
                embeddings
            )
        else:
            X_scaled = self.scaler.transform(
                embeddings
            )

        norms = np.linalg.norm(
            X_scaled,
            axis=1,
            keepdims=True
        )

        norms[norms == 0] = 1.0

        return (
            X_scaled / norms
        ).astype(np.float32)

    @staticmethod
    def _normalize_membership(membership):

        membership = np.asarray(
            membership,
            dtype=np.float32
        )

        row_sums = np.sum(
            membership,
            axis=1,
            keepdims=True
        )

        row_sums[row_sums == 0] = 1.0

        return (
            membership / row_sums
        ).astype(np.float32)

    def fit(self, embeddings):

        embeddings = self._validate_embeddings(
            embeddings
        )

        if embeddings.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) cannot be greater "
                f"than the number of samples "
                f"({embeddings.shape[0]})."
            )

        X_norm = self._prepare_embeddings(
            embeddings,
            fit=True
        )

        (
            self.centers,
            membership,
            _,
            _,
            _,
            _,
            self.fpc
        ) = fuzz.cluster.cmeans(
            X_norm.T,
            c=self.n_clusters,
            m=self.m,
            error=self.error,
            maxiter=self.max_iter,
            metric="cosine",
            seed=self.random_state,
            init=None,
        )

        self.centers = np.asarray(
            self.centers,
            dtype=np.float32
        )

        self.membership = (
            self._normalize_membership(
                membership.T
            )
        )

        return self

    def fit_transform(self, embeddings):

        self.fit(embeddings)

        return self.membership

    def transform(self, embeddings):

        if self.centers is None:
            raise RuntimeError(
                "Model has not been fitted."
            )

        X_norm = self._prepare_embeddings(
            embeddings,
            fit=False
        )

        (
            membership,
            _,
            _,
            _,
            _,
            _
        ) = fuzz.cluster.cmeans_predict(
            X_norm.T,
            self.centers,
            m=self.m,
            error=self.error,
            maxiter=self.max_iter,
            metric="cosine",
        )

        return self._normalize_membership(
            membership.T
        )

    def save(
        self,
        path="models/fuzzy_model.pkl"
    ):

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        joblib.dump(
            self,
            path
        )

    @staticmethod
    def load(
        path="models/fuzzy_model.pkl"
    ):

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Fuzzy clustering model not found: {path}"
            )

        return joblib.load(path)
