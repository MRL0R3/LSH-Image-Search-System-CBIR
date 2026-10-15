
import numpy as np
from typing import List, Tuple, Optional
import random


class AnnoyNode:


    def __init__(self, is_leaf=False):
        self.is_leaf = is_leaf
        self.left = None
        self.right = None
        self.hyperplane_normal = None
        self.hyperplane_offset = 0.0
        self.point_ids = []


class AnnoyTree:

    def __init__(self, dim: int, max_leaf_size: int = 10):
        self.dim = dim
        self.max_leaf_size = max_leaf_size
        self.root = None

    def _split_points(self, point_ids: List[int], vectors: dict) -> Tuple[List[int], List[int]]:
        if len(point_ids) <= self.max_leaf_size:
            return point_ids, []

        idx1, idx2 = random.sample(point_ids, 2)
        v1, v2 = vectors[idx1], vectors[idx2]

        normal = v2 - v1
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm

        midpoint = (v1 + v2) / 2
        offset = np.dot(normal, midpoint)

        left_ids, right_ids = [], []
        for pid in point_ids:
            if np.dot(normal, vectors[pid]) < offset:
                left_ids.append(pid)
            else:
                right_ids.append(pid)

        if not left_ids or not right_ids:
            mid = len(point_ids) // 2
            return point_ids[:mid], point_ids[mid:]

        return left_ids, right_ids, normal, offset

    def build(self, point_ids: List[int], vectors: dict):
        if len(point_ids) <= self.max_leaf_size:
            node = AnnoyNode(is_leaf=True)
            node.point_ids = point_ids
            return node

        result = self._split_points(point_ids, vectors)
        if len(result) == 2:
            node = AnnoyNode(is_leaf=True)
            node.point_ids = point_ids
            return node

        left_ids, right_ids, normal, offset = result

        node = AnnoyNode(is_leaf=False)
        node.hyperplane_normal = normal
        node.hyperplane_offset = offset

        # Recursive build
        node.left = self.build(left_ids, vectors)
        node.right = self.build(right_ids, vectors)

        return node

    def search(self, query: np.ndarray, node: Optional[AnnoyNode] = None) -> List[int]:
        if node is None:
            node = self.root

        if node.is_leaf:
            return node.point_ids

        if np.dot(query, node.hyperplane_normal) < node.hyperplane_offset:
            return self.search(query, node.left)
        else:
            return self.search(query, node.right)


class AnnoyIndex:


    def __init__(self, dim: int, n_trees: int = 10, max_leaf_size: int = 10):
        self.dim = dim
        self.n_trees = n_trees
        self.max_leaf_size = max_leaf_size

        self.trees = []
        self.vectors = {}
        self.built = False

    def add_vector(self, vec_id: int, vector: np.ndarray):
        if self.built:
            raise ValueError("Cannot add vectors after build()")
        self.vectors[vec_id] = vector.copy()

    def build(self):
        if self.built:
            return

        point_ids = list(self.vectors.keys())

        for _ in range(self.n_trees):
            tree = AnnoyTree(self.dim, self.max_leaf_size)
            tree.root = tree.build(point_ids, self.vectors)
            self.trees.append(tree)

        self.built = True

    def search(self, query: np.ndarray, k: int = 10,
               search_k: int = -1) -> List[Tuple[int, float]]:

        if not self.built:
            raise ValueError("Must call build() first")

        if search_k == -1:
            search_k = k * self.n_trees

        candidates = set()
        for tree in self.trees:
            candidates.update(tree.search(query))

        distances = []
        for cand_id in candidates:
            dist = np.linalg.norm(query - self.vectors[cand_id])
            distances.append((cand_id, dist))

        distances.sort(key=lambda x: x[1])
        return distances[:k]

    def get_stats(self) -> dict:
        return {
            'total_vectors': len(self.vectors),
            'n_trees': self.n_trees,
            'max_leaf_size': self.max_leaf_size,
            'built': self.built
        }
