import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.core.vector_db import VectorDatabase
except ImportError:
    from core.vector_db import VectorDatabase

db_path = SRC_DIR / "data" / "embeddings" / "caltech101_db.pkl"

db = VectorDatabase(
    dim=512,
    persist_path=str(db_path),
    use_lsh=True,
    lsh_params={'num_tables': 8, 'hash_size': 10, 'seed': 42}
)

if not db.vectors:
    print(f"Warning: Database at {db_path} is empty or could not be loaded.")
else:
    print(f"Loaded {len(db.vectors)} vectors from {db_path}")

target_id = "img_08050"
meta = db.get_metadata(target_id)
print(f"Metadata for {target_id}:", meta)
