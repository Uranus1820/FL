import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from typing import List, Tuple, Optional
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import linkage, fcluster
import warnings

warnings.filterwarnings('ignore')


class EnhancedDynamicGMMClusterer:
    """
    Enhanced GMM-based clustering with recent advances from 2024 research:
    - Gradient trajectory accumulation (CFLGT 2024)
    - Adaptive hierarchical clustering (RMPFD 2025)
    - Dynamic cluster number detection (FedRAC 2024)
    - Low-rank cosine similarity (FedAC 2024)
    """

    def __init__(self, args):
        self.args = args
        # History for gradient trajectory smoothing
        self.gradient_history = []
        self.max_history_length = getattr(args, 'trajectory_history', 5)

    def _cosine_distance(self, vec1, vec2):
        """
        Compute cosine distance (1 - Cosine Similarity).
        Range [0, 2], smaller is more similar.
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 1.0

        cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        return 1.0 - cosine_sim

    def _low_rank_similarity(self, vec1, vec2, rank=10):
        """
        Low-rank cosine similarity for efficient computation (FedAC 2024).
        Reduces dimensionality before similarity computation.

        Note: For small vectors (like 2-layer parameters), this automatically
        falls back to direct cosine distance, which is the right approach.
        """
        vec_dim = len(vec1)

        # For small vectors, direct computation is best and fastest
        # Low-rank is only beneficial for very high-dimensional vectors (100+)
        if vec_dim < 100:
            return self._cosine_distance(vec1, vec2)

        # For high-dimensional vectors, use adaptive rank
        # Rank should be much smaller than vector dimension
        adaptive_rank = min(rank, vec_dim // 10, 50)

        if adaptive_rank < 2:
            # If adaptive rank is too small, use direct computation
            return self._cosine_distance(vec1, vec2)

        # Project to lower rank for efficiency
        combined = np.vstack([vec1.reshape(1, -1), vec2.reshape(1, -1)])
        n_samples, n_features = combined.shape

        # Ensure n_components is valid: must be <= min(n_samples, n_features)
        max_components = min(n_samples, n_features, adaptive_rank)

        if max_components >= 2:
            try:
                pca_temp = PCA(n_components=max_components, random_state=42)
                combined_reduced = pca_temp.fit_transform(combined)
                return self._cosine_distance(combined_reduced[0], combined_reduced[1])
            except:
                # If PCA fails for any reason, fall back to direct computation
                return self._cosine_distance(vec1, vec2)

        # Fallback to direct computation
        return self._cosine_distance(vec1, vec2)

    def _compute_dunn_index(self, data, labels):
        """
        Compute Dunn Index for cluster validity (FedRAC 2024).
        Higher values indicate better clustering.
        """
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1:
            return 0.0

        # Compute cluster centers
        centers = []
        for label in unique_labels:
            cluster_data = data[labels == label]
            centers.append(np.mean(cluster_data, axis=0))
        centers = np.array(centers)

        # Min inter-cluster distance
        min_inter_dist = float('inf')
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.linalg.norm(centers[i] - centers[j])
                min_inter_dist = min(min_inter_dist, dist)

        # Max intra-cluster distance
        max_intra_dist = 0.0
        for label in unique_labels:
            cluster_data = data[labels == label]
            if len(cluster_data) > 1:
                center = centers[label]
                max_dist = np.max([np.linalg.norm(point - center)
                                   for point in cluster_data])
                max_intra_dist = max(max_intra_dist, max_dist)

        if max_intra_dist == 0:
            return 0.0

        return min_inter_dist / max_intra_dist

    def _gradient_trajectory_smoothing(self, current_vectors):
        """
        Smooth gradients using trajectory history (FedGTS 2025).
        Reduces oscillations and improves stability.
        """
        # Add current vectors to history
        self.gradient_history.append(current_vectors)
        if len(self.gradient_history) > self.max_history_length:
            self.gradient_history.pop(0)

        # If not enough history, return current vectors
        if len(self.gradient_history) < 2:
            return current_vectors

        # Compute smoothed vectors using exponential moving average
        smoothed_vectors = []
        alpha = 0.3  # Smoothing factor

        for i in range(len(current_vectors)):
            smoothed = current_vectors[i].copy()
            weight = alpha

            # Exponentially weighted average of historical gradients
            for hist_idx in range(len(self.gradient_history) - 2, -1, -1):
                if i < len(self.gradient_history[hist_idx]):
                    smoothed += weight * self.gradient_history[hist_idx][i]
                    weight *= alpha

            # Normalize
            norm = np.linalg.norm(smoothed)
            if norm > 0:
                smoothed = smoothed / norm

            smoothed_vectors.append(smoothed)

        return smoothed_vectors

    def _adaptive_hierarchical_clustering(self, data, vectors):
        """
        Adaptive hierarchical clustering (RMPFD 2025).
        Dynamically determines optimal cluster number.
        """
        n_samples = len(vectors)

        # Try different numbers of clusters
        best_k = 1
        best_score = -float('inf')

        max_k = min(n_samples, getattr(self.args, 'max_clusters', 10))
        min_k = max(2, getattr(self.args, 'min_clusters', 2))

        for k in range(min_k, max_k + 1):
            try:
                # Perform hierarchical clustering
                Z = linkage(data, method='ward')
                labels = fcluster(Z, k, criterion='maxclust') - 1

                # Compute Dunn index
                score = self._compute_dunn_index(data, labels)

                # Penalize too many or too few clusters
                cluster_penalty = abs(k - n_samples / 3) / n_samples
                adjusted_score = score - 0.1 * cluster_penalty

                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_k = k
            except:
                continue

        return best_k

    def _dynamic_gmm(self, vectors: List[np.ndarray]) -> Tuple[np.ndarray, List[int]]:
        """
        Enhanced GMM with gradient trajectory smoothing and adaptive clustering.
        """
        if not vectors:
            return np.array([]), []

        # Apply gradient trajectory smoothing
        smoothed_vectors = self._gradient_trajectory_smoothing(vectors)
        # Data preparation
        raw_data = np.vstack(smoothed_vectors)

        n_samples, n_features = raw_data.shape

        # PCA for dimensionality reduction
        use_pca = n_features > 50
        if use_pca:
            n_components = min(n_samples, 50)
            pca = PCA(n_components=n_components, random_state=42)
            data = pca.fit_transform(normalize(raw_data))
        else:
            data = normalize(raw_data)

        # Adaptive cluster number detection
        if getattr(self.args, 'auto_clusters', True):
            k_init = self._adaptive_hierarchical_clustering(data, vectors)
        else:
            k_init = n_samples

        # GMM initialization
        gmm = GaussianMixture(
            n_components=min(k_init, n_samples),
            covariance_type="diag",
            random_state=42,
            max_iter=100
        )

        try:
            gmm.fit(data)
            means = list(gmm.means_)
            weights = list(gmm.weights_)
        except:
            # Fallback to simple initialization
            means = [data[i] for i in range(min(k_init, n_samples))]
            weights = [1.0 / len(means)] * len(means)

        merged = True
        loop_count = 0
        max_loops = 20

        # Iterative merging with low-rank similarity
        while merged and loop_count < max_loops:
            merged = False
            loop_count += 1

            new_means = []
            new_weights = []
            used = set()

            min_dist_found = float('inf')

            for i in range(len(means)):
                if i in used:
                    continue

                group_indices = [i]

                for j in range(i + 1, len(means)):
                    if j in used:
                        continue

                    # Use low-rank similarity for efficiency
                    if getattr(self.args, 'use_low_rank', True):
                        dist = self._low_rank_similarity(means[i], means[j], rank=10)
                    else:
                        dist = self._cosine_distance(means[i], means[j])

                    if dist < min_dist_found:
                        min_dist_found = dist

                    # Adaptive threshold based on iteration
                    adaptive_threshold = self.args.epsilon_merge * (1 + 0.05 * loop_count)

                    if dist <= adaptive_threshold:
                        group_indices.append(j)
                        used.add(j)

                # Weighted average for merging
                w_sum = sum(weights[idx] for idx in group_indices)
                m_new = sum(weights[idx] * means[idx] for idx in group_indices) / w_sum

                # Normalize to unit sphere
                norm = np.linalg.norm(m_new)
                if norm > 0:
                    m_new = m_new / norm

                new_means.append(m_new)
                new_weights.append(w_sum)

                if len(group_indices) > 1:
                    merged = True

            means, weights = new_means, new_weights

        # Final assignment
        centers = np.vstack(means)
        assignments = []

        for vec in data:
            dists = [self._cosine_distance(vec, c) for c in centers]
            assignments.append(int(np.argmin(dists)))

        # Inverse PCA transform
        if use_pca:
            centers_original = pca.inverse_transform(centers)
        else:
            centers_original = centers

        return centers_original, assignments

    def cluster(self, vectors: List[np.ndarray]) -> Tuple[np.ndarray, List[int]]:
        """
        Main clustering interface.
        """
        return self._dynamic_gmm(vectors)

    def reset_history(self):
        """
        Reset gradient trajectory history.
        """
        self.gradient_history = []