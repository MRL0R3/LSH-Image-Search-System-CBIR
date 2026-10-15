import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Set
import time
import matplotlib.pyplot as plt
try:
    from sklearn.metrics import precision_score, recall_score
except ImportError:
    precision_score, recall_score = None, None
import os


class LSHIndex:


    def __init__(self, dim: int, num_tables: int = 5, hash_size: int = 8, seed: int = 42):

        self.dim = dim
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.projections = self._generate_projections()

        self.tables: List[Dict[str, List[str]]] = [{} for _ in range(num_tables)]

        logging.info(f"LSHIndex initialized with {num_tables} tables, {hash_size} bits each")

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('rng', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.rng = None

    def _generate_projections(self) -> List[np.ndarray]:
        projections = []
        for _ in range(self.num_tables):
            proj_matrix = self.rng.standard_normal((self.hash_size, self.dim))
            projections.append(proj_matrix)
        return projections

    def _hash_vector(self, vector: np.ndarray, proj_matrix: np.ndarray) -> str:

        projections = proj_matrix @ vector

        bits = (projections > 0).astype(int)

        return ''.join(str(bit) for bit in bits.flatten())

    def insert(self, vector_id: str, vector: np.ndarray) -> None:

        if vector.shape[0] != self.dim:
            raise ValueError(f"Vector dimension mismatch. Expected {self.dim}, got {vector.shape[0]}")

        for i, proj_matrix in enumerate(self.projections):
            hash_value = self._hash_vector(vector, proj_matrix)

            if hash_value not in self.tables[i]:
                self.tables[i][hash_value] = []
            self.tables[i][hash_value].append(vector_id)

    def delete(self, vector_id: str, vector: Optional[np.ndarray] = None) -> bool:
        """
        Delete a vector ID from all hash tables.

        If vector is provided, hashes directly into expected buckets for fast O(num_tables) removal.
        If vector is None or not found via direct hashing, falls back to scanning all buckets.

        Args:
            vector_id: The ID of the vector to delete.
            vector: Optional numpy array representing the vector, used for targeted bucket lookup.

        Returns:
            bool: True if vector_id was found and removed from at least one table, False otherwise.

        Raises:
            ValueError: If vector is provided but its dimension does not match self.dim.
        """
        if vector is not None:
            if vector.shape[0] != self.dim:
                raise ValueError(f"Vector dimension mismatch. Expected {self.dim}, got {vector.shape[0]}")

            removed = False
            for i, proj_matrix in enumerate(self.projections):
                hash_value = self._hash_vector(vector, proj_matrix)
                if hash_value in self.tables[i]:
                    bucket = self.tables[i][hash_value]
                    if vector_id in bucket:
                        self.tables[i][hash_value] = [vid for vid in bucket if vid != vector_id]
                        if not self.tables[i][hash_value]:
                            del self.tables[i][hash_value]
                        removed = True
            if removed:
                return True

        # Fallback full scan (when vector is None or vector did not match all stored buckets)
        found = False
        for i in range(self.num_tables):
            empty_hashes = []
            for hash_value, bucket in self.tables[i].items():
                if vector_id in bucket:
                    self.tables[i][hash_value] = [vid for vid in bucket if vid != vector_id]
                    found = True
                    if not self.tables[i][hash_value]:
                        empty_hashes.append(hash_value)
            for h in empty_hashes:
                del self.tables[i][h]

        return found

    def query(self, query_vector: np.ndarray, max_results: int = 100) -> List[str]:

        if query_vector.shape[0] != self.dim:
            raise ValueError(f"Query vector dimension mismatch. Expected {self.dim}, got {query_vector.shape[0]}")

        candidates: Set[str] = set()

        for i, proj_matrix in enumerate(self.projections):
            hash_value = self._hash_vector(query_vector, proj_matrix)

            if hash_value in self.tables[i]:
                candidates.update(self.tables[i][hash_value])

            if len(candidates) >= max_results * 2:
                break

        return list(candidates)[:max_results]

    def clear(self) -> None:
        self.tables = [{} for _ in range(self.num_tables)]
        logging.info("LSH index cleared")

    def get_stats(self) -> Dict[str, any]:
        stats = {
            'num_tables': self.num_tables,
            'hash_size': self.hash_size,
            'total_buckets': sum(len(table) for table in self.tables),
            'avg_bucket_size': np.mean([len(bucket) for table in self.tables for bucket in table.values()]) if
            self.tables[0] else 0,
            'max_bucket_size': max((len(bucket) for table in self.tables for bucket in table.values()), default=0)
        }
        return stats

    def benchmark(self, vector_db, query_vectors: List[Tuple[str, np.ndarray]], top_k: int = 10) -> Dict[str, any]:

        logging.info("Starting LSH benchmark...")

        results = {
            'lsh_query_times': [],
            'brute_query_times': [],
            'precisions': [],
            'recalls': [],
            'candidate_set_sizes': []
        }

        for query_id, query_vec in query_vectors:
            start_time = time.time()
            lsh_results = self.query(query_vec, max_results=top_k * 5)
            lsh_time = time.time() - start_time

            start_time = time.time()
            brute_results = self._brute_force_search(vector_db, query_vec, top_k)
            brute_time = time.time() - start_time

            true_positives = len(set(lsh_results[:top_k]).intersection(set(brute_results)))
            precision = true_positives / min(top_k, len(lsh_results)) if lsh_results else 0
            recall = true_positives / top_k if brute_results else 0

            results['lsh_query_times'].append(lsh_time)
            results['brute_query_times'].append(brute_time)
            results['precisions'].append(precision)
            results['recalls'].append(recall)
            results['candidate_set_sizes'].append(len(lsh_results))

        avg_results = {
            'avg_lsh_query_time': np.mean(results['lsh_query_times']),
            'avg_brute_query_time': np.mean(results['brute_query_times']),
            'avg_precision': np.mean(results['precisions']),
            'avg_recall': np.mean(results['recalls']),
            'avg_speedup': np.mean(results['brute_query_times']) / np.mean(results['lsh_query_times']) if np.mean(
                results['lsh_query_times']) > 0 else 0,
            'avg_candidate_set_size': np.mean(results['candidate_set_sizes']),
            'raw_results': results
        }

        logging.info(f"LSH Benchmark Results:")
        logging.info(f"  Avg LSH Query Time: {avg_results['avg_lsh_query_time']:.6f}s")
        logging.info(f"  Avg Brute Query Time: {avg_results['avg_brute_query_time']:.6f}s")
        logging.info(f"  Avg Speedup: {avg_results['avg_speedup']:.2f}x")
        logging.info(f"  Avg Precision@{top_k}: {avg_results['avg_precision']:.4f}")
        logging.info(f"  Avg Recall@{top_k}: {avg_results['avg_recall']:.4f}")

        return avg_results

    def _brute_force_search(self, vector_db, query_vector: np.ndarray, top_k: int) -> List[str]:
        similarities = []
        all_ids = vector_db.get_all_ids()

        for vec_id in all_ids:
            vector = vector_db.get_vector(vec_id)
            if vector is not None:
                sim = np.dot(query_vector, vector) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(vector) + 1e-8
                )
                similarities.append((vec_id, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)

        return [vec_id for vec_id, _ in similarities[:top_k]]

    def plot_performance(self, benchmark_results: Dict[str, any], output_dir: str = "results/plots/lsh"):
        os.makedirs(output_dir, exist_ok=True)

        raw = benchmark_results['raw_results']

        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(raw['lsh_query_times'])), raw['lsh_query_times'],
                    alpha=0.6, label='LSH', color='blue')
        plt.scatter(range(len(raw['brute_query_times'])), raw['brute_query_times'],
                    alpha=0.6, label='Brute-force', color='red')
        plt.axhline(y=benchmark_results['avg_lsh_query_time'], color='blue', linestyle='--', alpha=0.7)
        plt.axhline(y=benchmark_results['avg_brute_query_time'], color='red', linestyle='--', alpha=0.7)
        plt.title('Query Time Comparison: LSH vs Brute-force')
        plt.xlabel('Query Index')
        plt.ylabel('Time (seconds)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'query_times.png'), dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.scatter(raw['precisions'], raw['recalls'], alpha=0.7, c=raw['candidate_set_sizes'],
                    s=50, cmap='viridis', edgecolors='black')
        plt.colorbar(label='Candidate Set Size')
        plt.axhline(y=benchmark_results['avg_recall'], color='red', linestyle='--', alpha=0.7,
                    label=f'Avg Recall: {benchmark_results["avg_recall"]:.3f}')
        plt.axvline(x=benchmark_results['avg_precision'], color='blue', linestyle='--', alpha=0.7,
                    label=f'Avg Precision: {benchmark_results["avg_precision"]:.3f}')
        plt.title('Precision-Recall Trade-off for LSH')
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'precision_recall.png'), dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(raw['candidate_set_sizes'], bins=20, alpha=0.7, color='green', edgecolor='black')
        plt.axvline(x=benchmark_results['avg_candidate_set_size'], color='red', linestyle='--',
                    label=f'Average: {benchmark_results["avg_candidate_set_size"]:.1f}')
        plt.title('Distribution of Candidate Set Sizes')
        plt.xlabel('Number of Candidates')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'candidate_sizes.png'), dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Performance plots saved to {output_dir}")