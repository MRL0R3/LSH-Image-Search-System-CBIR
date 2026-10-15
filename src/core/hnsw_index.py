

import numpy as np
import heapq
from typing import List, Tuple, Set
import random


class HNSWIndex:


    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200, max_level: int = 5):
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.max_level = max_level


        self.graph = [{} for _ in range(max_level + 1)]

        self.vectors = {}

        self.entry_point = None
        self.entry_level = 0

        self.node_levels = {}

    def _distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return np.linalg.norm(v1 - v2)

    def _get_random_level(self) -> int:
        level = 0
        while random.random() < 0.5 and level < self.max_level:
            level += 1
        return level

    def _search_layer(self, query: np.ndarray, entry_points: Set[int],
                      num_closest: int, layer: int) -> List[Tuple[float, int]]:

        visited = set()
        candidates = []
        w = []

        for point in entry_points:
            dist = self._distance(query, self.vectors[point])
            heapq.heappush(candidates, (-dist, point))  # max heap
            heapq.heappush(w, (dist, point))  # min heap
            visited.add(point)

        while candidates:
            current_dist, current = heapq.heappop(candidates)
            current_dist = -current_dist

            if current_dist > w[0][0]:
                break

            neighbors = self.graph[layer].get(current, set())
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist = self._distance(query, self.vectors[neighbor])

                    if dist < w[0][0] or len(w) < num_closest:
                        heapq.heappush(candidates, (-dist, neighbor))
                        heapq.heappush(w, (dist, neighbor))

                        if len(w) > num_closest:
                            heapq.heappop(w)

        return sorted(w)

    def _connect_neighbors(self, node_id: int, neighbors: List[Tuple[float, int]],
                           layer: int):
        if node_id not in self.graph[layer]:
            self.graph[layer][node_id] = set()

        for _, neighbor_id in neighbors[:self.M]:
            if neighbor_id == node_id:
                continue
            self.graph[layer][node_id].add(neighbor_id)

            if neighbor_id not in self.graph[layer]:
                self.graph[layer][neighbor_id] = set()
            self.graph[layer][neighbor_id].add(node_id)

            if len(self.graph[layer][neighbor_id]) > self.M:
                neighbor_vec = self.vectors[neighbor_id]
                connections = list(self.graph[layer][neighbor_id])

                dists = [(self._distance(neighbor_vec, self.vectors[c]), c)
                         for c in connections]
                dists.sort()

                self.graph[layer][neighbor_id] = set(c for _, c in dists[:self.M])

        if len(self.graph[layer][node_id]) > self.M:
            node_vec = self.vectors[node_id]
            connections = list(self.graph[layer][node_id])
            dists = [(self._distance(node_vec, self.vectors[c]), c) for c in connections]
            dists.sort()
            self.graph[layer][node_id] = set(c for _, c in dists[:self.M])

    def add_vector(self, node_id: int, vector: np.ndarray):

        # If node already exists (update), disconnect old edges across all layers
        if node_id in self.vectors:
            for lc in range(len(self.graph)):
                if node_id in self.graph[lc]:
                    for neighbor_id in list(self.graph[lc][node_id]):
                        if neighbor_id in self.graph[lc]:
                            self.graph[lc][neighbor_id].discard(node_id)
                    self.graph[lc][node_id].clear()

            if self.entry_point == node_id:
                other_nodes = [nid for nid in self.vectors if nid != node_id and nid in self.node_levels]
                if other_nodes:
                    self.entry_point = max(other_nodes, key=lambda nid: self.node_levels.get(nid, 0))
                    self.entry_level = self.node_levels[self.entry_point]
                else:
                    self.entry_point = None
                    self.entry_level = 0

        self.vectors[node_id] = vector

        level = self._get_random_level()
        self.node_levels[node_id] = level

        if self.entry_point is None:
            self.entry_point = node_id
            self.entry_level = level
            for lc in range(level + 1):
                self.graph[lc][node_id] = set()
            return

        if self.entry_point == node_id:
            self.entry_level = level

        nearest = {self.entry_point}

        for lc in range(self.entry_level, level, -1):
            nearest = set(node for _, node in
                          self._search_layer(vector, nearest, 1, lc))

        for lc in range(level, -1, -1):
            candidates = self._search_layer(vector, nearest, self.ef_construction, lc)

            neighbors = candidates[:self.M]
            self._connect_neighbors(node_id, neighbors, lc)

            nearest = set(node for _, node in candidates)

        if level > self.entry_level:
            self.entry_point = node_id
            self.entry_level = level

    def search(self, query: np.ndarray, k: int = 10,
               ef: int = 50) -> List[Tuple[int, float]]:

        if self.entry_point is None:
            return []

        ef = max(ef, k)
        nearest = {self.entry_point}

        for lc in range(self.entry_level, 0, -1):
            nearest = set(node for _, node in
                          self._search_layer(query, nearest, 1, lc))


        candidates = self._search_layer(query, nearest, ef, 0)


        results = [(node_id, dist) for dist, node_id in candidates[:k]]
        return results

    def get_stats(self) -> dict:
        total_connections = sum(len(connections)
                                for layer in self.graph
                                for connections in layer.values())

        return {
            'total_nodes': len(self.vectors),
            'num_layers': len(self.graph),
            'entry_level': self.entry_level,
            'total_connections': total_connections,
            'avg_connections': total_connections / len(self.vectors) if self.vectors else 0,
            'M': self.M,
            'ef_construction': self.ef_construction
        }
