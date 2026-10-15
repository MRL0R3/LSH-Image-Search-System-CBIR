import time
import pickle
import numpy as np
import os
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
from threading import RLock

DEFAULT_PERSIST_PATH = str(Path(__file__).resolve().parent.parent / "data" / "embeddings" / "vector_db.pkl")

try:
    from .lsh_index import LSHIndex
    from .hnsw_index import HNSWIndex
    HNSW_AVAILABLE = True
except ImportError:
    try:
        from core.lsh_index import LSHIndex
        from core.hnsw_index import HNSWIndex
        HNSW_AVAILABLE = True
    except ImportError:
        try:
            from src.core.lsh_index import LSHIndex
            from src.core.hnsw_index import HNSWIndex
            HNSW_AVAILABLE = True
        except ImportError as e:
            try:
                from .lsh_index import LSHIndex
            except ImportError:
                try:
                    from core.lsh_index import LSHIndex
                except ImportError:
                    from src.core.lsh_index import LSHIndex
            HNSW_AVAILABLE = False
            logging.warning(f"HNSW module could not be loaded ({e}). HNSW indexing will be disabled.")


class VectorDatabase:
    def __init__(self, dim: int, persist_path: Optional[str] = None,
                 use_lsh: bool = True, use_hnsw: bool = False,
                 lsh_params: dict = None, hnsw_params: dict = None):
        self.dim = dim
        self.persist_path = persist_path if persist_path is not None else DEFAULT_PERSIST_PATH
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.use_lsh = use_lsh
        self.use_hnsw = use_hnsw and HNSW_AVAILABLE

        self.lock = RLock()

        default_lsh_params = {
            'num_tables': 5,
            'hash_size': 8,
            'seed': 42
        }
        self.lsh_params = {**default_lsh_params, **(lsh_params or {})}

        default_hnsw_params = {
            'M': 16,
            'ef_construction': 200,
            'max_level': 5
        }
        self.hnsw_params = {**default_hnsw_params, **(hnsw_params or {})}

        self.lsh_index = LSHIndex(
            dim=dim,
            num_tables=self.lsh_params['num_tables'],
            hash_size=self.lsh_params['hash_size'],
            seed=self.lsh_params['seed']
        ) if use_lsh else None

        self.hnsw_index = None
        if self.use_hnsw:
            if HNSW_AVAILABLE:
                self.hnsw_index = HNSWIndex(
                    dim=dim,
                    M=self.hnsw_params['M'],
                    ef_construction=self.hnsw_params['ef_construction'],
                    max_level=self.hnsw_params['max_level']
                )
            else:
                self.use_hnsw = False

        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        self.load_from_disk()

    def add_vector(self, vector_id: str, vector: np.ndarray, metadata: Dict[str, Any] = None) -> bool:
        with self.lock:
            if vector_id in self.vectors:
                return False

            if vector.shape[0] != self.dim:
                return False

            self.vectors[vector_id] = vector.astype(np.float32)
            self.metadata[vector_id] = metadata or {}

            if self.use_lsh and self.lsh_index is not None:
                self.lsh_index.insert(vector_id, vector)

            if self.use_hnsw and self.hnsw_index is not None:
                numeric_id = self._get_or_create_numeric_id(vector_id)
                self.hnsw_index.add_vector(numeric_id, vector)

            return True

    def find_similar(self, query_vector: np.ndarray, top_k: int = 10,
                     use_lsh: Optional[bool] = None, use_hnsw: bool = False,
                     lsh_max_candidates: int = 100, hnsw_ef: int = 50) -> List[Tuple[str, float]]:
        with self.lock:
            if query_vector.shape[0] != self.dim:
                raise ValueError(f"Query vector dimension mismatch. Expected {self.dim}, got {query_vector.shape[0]}")

            if use_hnsw and self.use_hnsw and self.hnsw_index is not None:
                return self._search_with_hnsw(query_vector, top_k, hnsw_ef)

            if use_lsh is None:
                use_lsh = self.use_lsh

            candidate_ids = self.get_all_ids()

            if use_lsh and self.use_lsh and self.lsh_index is not None:
                candidate_ids = self.lsh_index.query(query_vector, max_results=lsh_max_candidates)

            similarities = []
            for vec_id in candidate_ids:
                vector = self.vectors.get(vec_id)
                if vector is not None:
                    sim = np.dot(query_vector, vector) / (
                            np.linalg.norm(query_vector) * np.linalg.norm(vector) + 1e-8
                    )
                    similarities.append((vec_id, sim))

            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]

    def _search_with_hnsw(self, query_vector: np.ndarray, top_k: int, ef: int) -> List[Tuple[str, float]]:
        if not self.vectors:
            return []

        target_k = min(len(self.vectors), top_k)
        total_graph_nodes = len(self.hnsw_index.vectors) if self.hnsw_index else len(self.vectors)
        deleted_count = max(0, total_graph_nodes - len(self.vectors))

        fetch_k = min(total_graph_nodes, max(target_k * 2, target_k + deleted_count))
        search_ef = max(ef, fetch_k + deleted_count)

        hnsw_results = self.hnsw_index.search(query_vector, k=fetch_k, ef=search_ef)
        results = []
        for numeric_id, distance in hnsw_results:
            string_id = self._numeric_to_string_id(numeric_id)
            if string_id and string_id in self.vectors:
                similarity = 1.0 / (1.0 + distance)
                results.append((string_id, similarity))
                if len(results) >= target_k:
                    break

        # Fallback safeguard: If graph search returned fewer than target_k active candidates
        # (e.g. extreme deletion churn or clustered deleted nodes),
        # complement with exact linear scan over remaining active vectors.
        if len(results) < target_k:
            returned_ids = set(r[0] for r in results)
            remaining = []
            for vec_id, vec in self.vectors.items():
                if vec_id not in returned_ids:
                    dist = np.linalg.norm(query_vector - vec)
                    sim = 1.0 / (1.0 + dist)
                    remaining.append((vec_id, sim))
            remaining.sort(key=lambda x: x[1], reverse=True)
            results.extend(remaining[:target_k - len(results)])

        return results

    def reindex_hnsw(self):
        """Rebuild HNSW index from scratch using active vectors, purging all tombstoned nodes."""
        with self.lock:
            if self.use_hnsw:
                self._rebuild_hnsw_index()

    def _get_or_create_numeric_id(self, string_id: str) -> int:
        if not hasattr(self, '_id_mapping'):
            self._id_mapping = {}
            self._reverse_mapping = {}
            self._next_id = 0

        if string_id not in self._id_mapping:
            self._id_mapping[string_id] = self._next_id
            self._reverse_mapping[self._next_id] = string_id
            self._next_id += 1

        return self._id_mapping[string_id]

    def _numeric_to_string_id(self, numeric_id: int) -> Optional[str]:
        if hasattr(self, '_reverse_mapping'):
            return self._reverse_mapping.get(numeric_id)
        return None

    def benchmark_search_methods(self, test_queries: List[Tuple[str, np.ndarray]],
                                 top_k: int = 10) -> Dict[str, Any]:
        with self.lock:
            results = {
                'brute_force': {'times': [], 'results': []},
                'lsh': {'times': [], 'results': []},
                'hnsw': {'times': [], 'results': []}
            }

            for query_id, query_vec in test_queries:
                start_time = time.time()
                results_brute = self.find_similar(query_vec, top_k=top_k, use_lsh=False, use_hnsw=False)
                results['brute_force']['times'].append(time.time() - start_time)
                results['brute_force']['results'].append((query_id, results_brute))

            if self.use_lsh and self.lsh_index is not None:
                for query_id, query_vec in test_queries:
                    start_time = time.time()
                    results_lsh = self.find_similar(query_vec, top_k=top_k, use_lsh=True, use_hnsw=False)
                    results['lsh']['times'].append(time.time() - start_time)
                    results['lsh']['results'].append((query_id, results_lsh))

            if self.use_hnsw and self.hnsw_index is not None:
                for query_id, query_vec in test_queries:
                    start_time = time.time()
                    results_hnsw = self.find_similar(query_vec, top_k=top_k, use_hnsw=True)
                    results['hnsw']['times'].append(time.time() - start_time)
                    results['hnsw']['results'].append((query_id, results_hnsw))

            self._print_benchmark_stats(results, top_k)
            return results

    def _print_benchmark_stats(self, results: Dict, top_k: int):
        avg_brute_time = np.mean(results['brute_force']['times']) if results['brute_force']['times'] else 0

        logging.info(f"Brute-force: Avg Time: {avg_brute_time * 1000:.2f} ms")

        if results['lsh']['times']:
            avg_lsh_time = np.mean(results['lsh']['times'])
            lsh_precision = self._calculate_precision(results, 'lsh', top_k)
            logging.info(f"LSH: Avg Time: {avg_lsh_time * 1000:.2f} ms | Precision: {lsh_precision * 100:.2f}%")

        if results['hnsw']['times']:
            avg_hnsw_time = np.mean(results['hnsw']['times'])
            hnsw_precision = self._calculate_precision(results, 'hnsw', top_k)
            logging.info(f"HNSW: Avg Time: {avg_hnsw_time * 1000:.2f} ms | Precision: {hnsw_precision * 100:.2f}%")

    def _calculate_precision(self, results: Dict, method: str, top_k: int) -> float:
        precisions = []
        num_queries = len(results['brute_force']['results'])
        for i in range(num_queries):
            brute_results = set([res[0] for res in results['brute_force']['results'][i][1][:top_k]])
            method_results = set([res[0] for res in results[method]['results'][i][1][:top_k]])
            if method_results:
                precision = len(method_results.intersection(brute_results)) / len(method_results)
                precisions.append(precision)
        return np.mean(precisions) if precisions else 0.0

    def save_to_disk(self):
        with self.lock:
            if self.persist_path is None:
                return
            try:
                os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
                data_to_save = {
                    'vectors': self.vectors,
                    'metadata': self.metadata,
                    'dim': self.dim,
                    'use_lsh': self.use_lsh,
                    'use_hnsw': self.use_hnsw,
                    'lsh_params': self.lsh_params,
                    'hnsw_params': self.hnsw_params
                }
                if self.use_lsh and self.lsh_index is not None:
                    data_to_save['lsh_index'] = self.lsh_index
                if self.use_hnsw and self.hnsw_index is not None:
                    data_to_save['hnsw_index'] = self.hnsw_index
                    data_to_save['id_mapping'] = getattr(self, '_id_mapping', {})
                    data_to_save['reverse_mapping'] = getattr(self, '_reverse_mapping', {})
                    data_to_save['next_id'] = getattr(self, '_next_id', 0)

                with open(self.persist_path, 'wb') as f:
                    pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                logging.error(f"Save failed: {e}")

    def load_from_disk(self):
        with self.lock:
            if self.persist_path is None or not os.path.exists(self.persist_path):
                return
            try:
                # Ensure transparent compatibility between NumPy 1.x and NumPy 2.x pickles
                try:
                    import numpy._core
                except ImportError:
                    import sys as _sys, numpy.core as _core
                    _sys.modules['numpy._core'] = _core
                    _sys.modules['numpy._core.multiarray'] = _core.multiarray
                    _sys.modules['numpy._core._multiarray_umath'] = getattr(_core, '_multiarray_umath', _core.multiarray)
                    _sys.modules['numpy._core.umath'] = getattr(_core, 'umath', _core.multiarray)
                try:
                    import numpy.core
                except ImportError:
                    import sys as _sys, numpy._core as _ncore
                    _sys.modules['numpy.core'] = _ncore
                    _sys.modules['numpy.core.multiarray'] = _ncore.multiarray

                with open(self.persist_path, 'rb') as f:
                    data = pickle.load(f)
                self.vectors = data.get('vectors', {})
                self.metadata = data.get('metadata', {})
                self.dim = data.get('dim', self.dim)
                self.use_lsh = data.get('use_lsh', self.use_lsh)
                self.use_hnsw = data.get('use_hnsw', self.use_hnsw)
                self.lsh_params = data.get('lsh_params', self.lsh_params)
                self.hnsw_params = data.get('hnsw_params', self.hnsw_params)

                if self.use_lsh:
                    if 'lsh_index' in data and data['lsh_index'] is not None:
                        self.lsh_index = data['lsh_index']
                    else:
                        self._rebuild_lsh_index()

                if self.use_hnsw and HNSW_AVAILABLE:
                    if 'hnsw_index' in data and data['hnsw_index'] is not None:
                        self.hnsw_index = data['hnsw_index']
                        self._id_mapping = data.get('id_mapping', {})
                        self._reverse_mapping = data.get('reverse_mapping', {})
                        self._next_id = data.get('next_id', 0)
                    else:
                        self._rebuild_hnsw_index()
            except Exception as e:
                logging.error(f"Load failed: {e}")
                raise

    def _rebuild_lsh_index(self):
        self.lsh_index = LSHIndex(
            dim=self.dim,
            num_tables=self.lsh_params['num_tables'],
            hash_size=self.lsh_params['hash_size'],
            seed=self.lsh_params['seed']
        )
        for vec_id, vector in self.vectors.items():
            self.lsh_index.insert(vec_id, vector)

    def _rebuild_hnsw_index(self):
        self.hnsw_index = HNSWIndex(
            dim=self.dim,
            M=self.hnsw_params['M'],
            ef_construction=self.hnsw_params['ef_construction'],
            max_level=self.hnsw_params['max_level']
        )
        self._id_mapping = {}
        self._reverse_mapping = {}
        self._next_id = 0
        for vec_id, vector in self.vectors.items():
            numeric_id = self._get_or_create_numeric_id(vec_id)
            self.hnsw_index.add_vector(numeric_id, vector)

    def get_all_ids(self) -> List[str]:
        with self.lock:
            return list(self.vectors.keys())

    def get_vector(self, vector_id: str) -> Optional[np.ndarray]:
        with self.lock:
            return self.vectors.get(vector_id)

    def get_metadata(self, vector_id: str) -> Dict[str, Any]:
        with self.lock:
            return self.metadata.get(vector_id, {})

    def clear_database(self):
        with self.lock:
            self.vectors.clear()
            self.metadata.clear()
            if self.use_lsh: self._rebuild_lsh_index()
            if self.use_hnsw: self._rebuild_hnsw_index()

    def update_vector(self, vector_id: str, vector: np.ndarray, metadata: Dict[str, Any] = None) -> bool:
        with self.lock:
            if vector_id not in self.vectors:
                return False
            if vector.shape[0] != self.dim:
                return False
            old_vector = self.vectors[vector_id]
            self.vectors[vector_id] = vector.astype(np.float32)
            if metadata is not None:
                if vector_id not in self.metadata:
                    self.metadata[vector_id] = {}
                self.metadata[vector_id].update(metadata)
            if self.use_lsh and self.lsh_index is not None:
                self.lsh_index.delete(vector_id, old_vector)
                self.lsh_index.insert(vector_id, vector)
            if self.use_hnsw and self.hnsw_index is not None:
                numeric_id = self._get_or_create_numeric_id(vector_id)
                self.hnsw_index.add_vector(numeric_id, vector)
            return True

    def delete_vector(self, vector_id: str) -> bool:
        with self.lock:
            if vector_id not in self.vectors:
                return False
            old_vector = self.vectors[vector_id]
            del self.vectors[vector_id]
            if vector_id in self.metadata:
                del self.metadata[vector_id]
            if self.use_lsh and self.lsh_index is not None:
                self.lsh_index.delete(vector_id, old_vector)
            if self.use_hnsw and self.hnsw_index is not None:
                if hasattr(self, '_id_mapping') and vector_id in self._id_mapping:
                    numeric_id = self._id_mapping.pop(vector_id)
                    if hasattr(self, '_reverse_mapping'):
                        self._reverse_mapping.pop(numeric_id, None)
            return True

    def get_database_stats(self) -> Dict[str, Any]:
        with self.lock:
            stats = {
                'total_vectors': len(self.vectors),
                'dimension': self.dim,
                'use_lsh': self.use_lsh,
                'use_hnsw': self.use_hnsw,
                'persist_path': self.persist_path,
                'thread_safe': True
            }
            if self.use_lsh and self.lsh_index is not None:
                stats['lsh_stats'] = self.lsh_index.get_stats()
            if self.use_hnsw and self.hnsw_index is not None:
                stats['hnsw_stats'] = self.hnsw_index.get_stats()
            return stats