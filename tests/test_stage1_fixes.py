import sys
import os
import unittest
from pathlib import Path
import numpy as np

# Ensure project root and src/ are in sys.path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.core.lsh_index import LSHIndex
from src.core.vector_db import VectorDatabase, HNSW_AVAILABLE
from src.core.hnsw_index import HNSWIndex


import tempfile

class TestStage1Fixes(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(12345)
        self.dim = 16
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_db_path = os.path.join(self.temp_dir.name, "temp_test_db.pkl")

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except OSError:
            pass

    def test_01_lsh_delete_with_vector_and_scan_fallback(self):
        """Test LSHIndex delete with direct cached vector and fallback full scan."""
        lsh = LSHIndex(dim=self.dim, num_tables=4, hash_size=6, seed=42)
        v1 = self.rng.standard_normal(self.dim).astype(np.float32)
        v2 = self.rng.standard_normal(self.dim).astype(np.float32)
        v3 = self.rng.standard_normal(self.dim).astype(np.float32)

        lsh.insert("v1", v1)
        lsh.insert("v2", v2)
        lsh.insert("v3", v3)

        # Confirm v1 is retrievable
        candidates = lsh.query(v1, max_results=10)
        self.assertIn("v1", candidates)

        # Delete v1 using fast-path (vector provided)
        del_result = lsh.delete("v1", v1)
        self.assertTrue(del_result)

        # Confirm v1 is no longer in any candidate query
        candidates_after = lsh.query(v1, max_results=10)
        self.assertNotIn("v1", candidates_after)

        # Double delete returns False
        self.assertFalse(lsh.delete("v1", v1))

        # Delete nonexistent ID returns False
        self.assertFalse(lsh.delete("nonexistent_id", v2))

        # Delete v2 using fallback full scan (vector=None)
        del_v2_result = lsh.delete("v2")
        self.assertTrue(del_v2_result)
        candidates_v2 = lsh.query(v2, max_results=10)
        self.assertNotIn("v2", candidates_v2)

        # Dimension validation in delete
        wrong_dim = np.zeros(self.dim + 2, dtype=np.float32)
        with self.assertRaises(ValueError):
            lsh.delete("v3", wrong_dim)

    def test_02_lsh_update_no_duplicate_entries(self):
        """Test updating an entry in LSHIndex leaves exactly 1 entry per table."""
        lsh = LSHIndex(dim=self.dim, num_tables=4, hash_size=6, seed=42)
        v1_old = self.rng.standard_normal(self.dim).astype(np.float32)
        v1_new = self.rng.standard_normal(self.dim).astype(np.float32)

        lsh.insert("v1", v1_old)
        # Count total entries across all tables
        initial_count = sum(bucket.count("v1") for table in lsh.tables for bucket in table.values())
        self.assertEqual(initial_count, lsh.num_tables)

        # Perform update (delete old, insert new)
        lsh.delete("v1", v1_old)
        lsh.insert("v1", v1_new)

        updated_count = sum(bucket.count("v1") for table in lsh.tables for bucket in table.values())
        self.assertEqual(updated_count, lsh.num_tables)

    def test_03_vector_db_deletion_and_synchronization(self):
        """Test VectorDatabase deletes synchronize vector store, LSH, HNSW, and ID mappings."""
        db = VectorDatabase(
            dim=self.dim,
            persist_path=self.temp_db_path,
            use_lsh=True,
            use_hnsw=True,
            lsh_params={'num_tables': 4, 'hash_size': 6, 'seed': 42}
        )
        db.clear_database()

        v1 = self.rng.standard_normal(self.dim).astype(np.float32)
        v2 = self.rng.standard_normal(self.dim).astype(np.float32)
        meta1 = {"label": "item_1"}
        meta2 = {"label": "item_2"}

        db.add_vector("v1", v1, meta1)
        db.add_vector("v2", v2, meta2)

        self.assertIn("v1", db.vectors)
        self.assertEqual(db.get_metadata("v1")["label"], "item_1")
        self.assertIn("v1", db._id_mapping)
        numeric_id_v1 = db._id_mapping["v1"]
        self.assertEqual(db._reverse_mapping[numeric_id_v1], "v1")

        # Delete v1
        self.assertTrue(db.delete_vector("v1"))

        # Verify main store
        self.assertNotIn("v1", db.vectors)
        self.assertIsNone(db.get_vector("v1"))
        self.assertEqual(db.get_metadata("v1"), {})

        # Verify ID mappings purged
        self.assertNotIn("v1", db._id_mapping)
        self.assertNotIn(numeric_id_v1, db._reverse_mapping)

        # Verify LSH query never returns deleted vector
        lsh_res = db.find_similar(v1, top_k=5, use_lsh=True, use_hnsw=False)
        lsh_ids = [res[0] for res in lsh_res]
        self.assertNotIn("v1", lsh_ids)

        # Verify HNSW search never returns deleted vector
        hnsw_res = db.find_similar(v1, top_k=5, use_hnsw=True)
        hnsw_ids = [res[0] for res in hnsw_res]
        self.assertNotIn("v1", hnsw_ids)

        # Verify brute-force search never returns deleted vector
        brute_res = db.find_similar(v1, top_k=5, use_lsh=False, use_hnsw=False)
        brute_ids = [res[0] for res in brute_res]
        self.assertNotIn("v1", brute_ids)

        # Deleting nonexistent or already deleted ID returns False
        self.assertFalse(db.delete_vector("v1"))
        self.assertFalse(db.delete_vector("nonexistent"))

    def test_04_vector_db_update_synchronization(self):
        """Test VectorDatabase update modifies vectors, metadata, and keeps mappings clean."""
        db = VectorDatabase(
            dim=self.dim,
            persist_path=self.temp_db_path,
            use_lsh=True,
            use_hnsw=True,
            lsh_params={'num_tables': 4, 'hash_size': 6, 'seed': 42}
        )
        db.clear_database()

        v1 = self.rng.standard_normal(self.dim).astype(np.float32)
        v1_updated = self.rng.standard_normal(self.dim).astype(np.float32)
        db.add_vector("v1", v1, {"tag": "v1_original"})

        num_id_before = db._id_mapping["v1"]

        # Update vector and metadata
        self.assertTrue(db.update_vector("v1", v1_updated, {"tag": "v1_updated", "version": 2}))

        # Confirm ID mapping was preserved without creating duplicates
        self.assertEqual(db._id_mapping["v1"], num_id_before)
        self.assertEqual(db._reverse_mapping[num_id_before], "v1")
        self.assertEqual(len(db._id_mapping), 1)
        self.assertEqual(len(db._reverse_mapping), 1)

        # Confirm data and metadata updated
        np.testing.assert_array_equal(db.get_vector("v1"), v1_updated.astype(np.float32))
        meta = db.get_metadata("v1")
        self.assertEqual(meta["tag"], "v1_updated")
        self.assertEqual(meta["version"], 2)

        # Update nonexistent vector returns False
        self.assertFalse(db.update_vector("ghost", v1_updated))

        # Update with wrong dimension returns False
        wrong_dim = np.zeros(self.dim + 5, dtype=np.float32)
        self.assertFalse(db.update_vector("v1", wrong_dim))

    def test_05_dimension_validation(self):
        """Test dimension validation consistency across add and find operations."""
        db = VectorDatabase(dim=self.dim, persist_path=self.temp_db_path, use_lsh=True)
        db.clear_database()

        wrong_dim = np.zeros(self.dim + 3, dtype=np.float32)
        self.assertFalse(db.add_vector("bad", wrong_dim))

        with self.assertRaises(ValueError):
            db.find_similar(wrong_dim)

    def test_06_core_imports_and_hnsw_availability(self):
        """Verify HNSW is available and core modules import cleanly."""
        self.assertTrue(HNSW_AVAILABLE, "HNSWIndex should be available in core modules")

    def test_07_hnsw_starvation_after_heavy_deletion(self):
        """Verify HNSW search never starves when many nodes are deleted."""
        db = VectorDatabase(
            dim=self.dim,
            persist_path=self.temp_db_path,
            use_lsh=False,
            use_hnsw=True
        )
        db.clear_database()

        # Insert 20 vectors
        for i in range(20):
            vec = self.rng.standard_normal(self.dim).astype(np.float32)
            db.add_vector(f"v_{i}", vec, {"idx": i})

        # Delete 18 vectors (90% deletion rate)
        deleted_keys = [f"v_{i}" for i in range(18)]
        for k in deleted_keys:
            self.assertTrue(db.delete_vector(k))

        self.assertEqual(len(db.vectors), 2)

        # Query with top_k=5; active count is 2, so exactly 2 results must be returned
        q_vec = self.rng.standard_normal(self.dim).astype(np.float32)
        results = db.find_similar(q_vec, top_k=5, use_hnsw=True)
        self.assertEqual(len(results), 2, f"Expected 2 results, got {len(results)}")

        result_ids = [r[0] for r in results]
        for k in deleted_keys:
            self.assertNotIn(k, result_ids)
        self.assertIn("v_18", result_ids)
        self.assertIn("v_19", result_ids)

    def test_08_hnsw_repeated_updates_invariants(self):
        """Verify repeated vector updates maintain graph degree limits and clean mappings."""
        db = VectorDatabase(
            dim=self.dim,
            persist_path=self.temp_db_path,
            use_lsh=True,
            use_hnsw=True
        )
        db.clear_database()

        # Insert 10 vectors
        for i in range(10):
            vec = self.rng.standard_normal(self.dim).astype(np.float32)
            db.add_vector(f"v_{i}", vec, {"version": 1})

        # Update v_0 repeatedly 5 times with distinct vectors
        for step in range(2, 7):
            new_vec = self.rng.standard_normal(self.dim).astype(np.float32)
            self.assertTrue(db.update_vector("v_0", new_vec, {"version": step}))

        # Verify mapping consistency
        self.assertEqual(len(db._id_mapping), 10)
        self.assertEqual(len(db._reverse_mapping), 10)
        self.assertEqual(db.get_metadata("v_0")["version"], 6)

        # Check degree invariant at layer 0: node's neighbor count <= M
        num_id_0 = db._id_mapping["v_0"]
        degree_layer_0 = len(db.hnsw_index.graph[0].get(num_id_0, set()))
        self.assertLessEqual(degree_layer_0, db.hnsw_index.M)

    def test_09_save_and_reload_mutated_database(self):
        """Verify persistence round-trip with insertions, deletions, and updates."""
        db = VectorDatabase(
            dim=self.dim,
            persist_path=self.temp_db_path,
            use_lsh=True,
            use_hnsw=True
        )
        db.clear_database()

        # Add 5 items
        for i in range(5):
            vec = self.rng.standard_normal(self.dim).astype(np.float32)
            db.add_vector(f"item_{i}", vec, {"label": f"label_{i}"})

        # Update item_0
        updated_vec = self.rng.standard_normal(self.dim).astype(np.float32)
        db.update_vector("item_0", updated_vec, {"label": "label_0_updated"})

        # Delete item_1 and item_3
        db.delete_vector("item_1")
        db.delete_vector("item_3")

        self.assertEqual(len(db.vectors), 3)

        # Save to disk
        db.save_to_disk()
        self.assertTrue(os.path.exists(self.temp_db_path))

        # Reload in a new instance
        db_loaded = VectorDatabase(
            dim=self.dim,
            persist_path=self.temp_db_path,
            use_lsh=True,
            use_hnsw=True
        )
        db_loaded.load_from_disk()
        self.assertEqual(len(db_loaded.vectors), 3)
        self.assertNotIn("item_1", db_loaded.vectors)
        self.assertNotIn("item_3", db_loaded.vectors)
        self.assertEqual(db_loaded.get_metadata("item_0")["label"], "label_0_updated")

        # Test search across LSH and HNSW on loaded database
        res_hnsw = db_loaded.find_similar(updated_vec, top_k=5, use_hnsw=True)
        self.assertEqual(len(res_hnsw), 3)
        self.assertEqual(res_hnsw[0][0], "item_0")

        res_lsh = db_loaded.find_similar(updated_vec, top_k=5, use_lsh=True, use_hnsw=False)
        self.assertNotIn("item_1", [r[0] for r in res_lsh])
        self.assertNotIn("item_3", [r[0] for r in res_lsh])

    def test_10_delete_and_reinsert_same_id(self):
        """Verify deleting an ID and reinserting it works cleanly without stale state."""
        db = VectorDatabase(
            dim=self.dim,
            persist_path=self.temp_db_path,
            use_lsh=True,
            use_hnsw=True
        )
        db.clear_database()

        v_init = self.rng.standard_normal(self.dim).astype(np.float32)
        db.add_vector("item_x", v_init, {"phase": "first"})

        self.assertIn("item_x", db.vectors)
        self.assertTrue(db.delete_vector("item_x"))
        self.assertNotIn("item_x", db.vectors)

        v_new = self.rng.standard_normal(self.dim).astype(np.float32)
        self.assertTrue(db.add_vector("item_x", v_new, {"phase": "reinserted"}))

        self.assertEqual(len(db.vectors), 1)
        self.assertEqual(db.get_metadata("item_x")["phase"], "reinserted")

        # Verify search returns item_x
        results = db.find_similar(v_new, top_k=3, use_hnsw=True)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "item_x")


if __name__ == "__main__":
    unittest.main()
