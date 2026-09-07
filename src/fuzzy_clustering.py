import joblib
import numpy as np
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler


class FuzzyClustering:

    def __init__(self, n_clusters=5, m=1.5, max_iter=300, error=1e-5, random_state=42):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.error = error
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.centers = None
        self.membership = None
        self.fpc = None

    def fit(self, embeddings):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # استانداردسازی اولیه (اختیاری)
        X_scaled = self.scaler.fit_transform(embeddings)

        # نرمال‌سازی برداری (هر سطر طول ۱) برای استفاده از فاصله کسینوسی
        norms = np.linalg.norm(X_scaled, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_norm = X_scaled / norms

        # اجرای FCM با معیار کسینوسی
        # نکته: skfuzzy.cmeans داده را به شکل (n_features, n_samples) می‌پذیرد
        self.centers, membership, _, _, _, _, self.fpc = fuzz.cluster.cmeans(
            X_norm.T,                     # (n_features, n_samples)
            c=self.n_clusters,
            m=self.m,
            error=self.error,
            maxiter=self.max_iter,
            metric='cosine',              # فاصله کسینوسی
            seed=self.random_state,
            init=None,
        )

        mem = membership.T.astype(np.float32)   # (n_samples, n_clusters)
        self.membership = mem / np.sum(mem, axis=1, keepdims=True)

        return self

    def fit_transform(self, embeddings):
        self.fit(embeddings)
        return self.membership

    def transform(self, embeddings):
        if self.centers is None:
            raise RuntimeError("Model has not been fitted.")

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        X_scaled = self.scaler.transform(embeddings)

        norms = np.linalg.norm(X_scaled, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_norm = X_scaled / norms

        membership, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
            X_norm.T,
            self.centers,
            m=self.m,
            error=self.error,
            maxiter=self.max_iter,
            metric='cosine',
        )

        mem = membership.T.astype(np.float32)
        return mem / np.sum(mem, axis=1, keepdims=True)

    def save(self, path="models/fuzzy_model.pkl"):
        joblib.dump(self, path)

    @staticmethod
    def load(path="models/fuzzy_model.pkl"):
        return joblib.load(path)
