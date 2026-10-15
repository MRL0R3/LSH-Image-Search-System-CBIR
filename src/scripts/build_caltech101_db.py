
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.core.vector_db import VectorDatabase
except ImportError:
    from core.vector_db import VectorDatabase


def main():
    data_dir = SRC_DIR / "data" / "caltech101"
    emb_path = data_dir / "caltech101_embeddings.npy"
    ids_path = data_dir / "caltech101_image_ids.npy"

    print("Loading embeddings from:", emb_path)
    print("Loading image ids from:", ids_path)

    embeddings = np.load(emb_path)
    image_ids = np.load(ids_path)

    print("embeddings shape:", embeddings.shape)
    print("image_ids shape:", image_ids.shape)
    print("example id:", image_ids[0])



    db_path = SRC_DIR / "data" / "embeddings" / "caltech101_db.pkl"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        try:
            db_path.unlink()
        except OSError:
            pass

    dim = embeddings.shape[1]

    vector_db = VectorDatabase(
        dim=dim,
        persist_path=str(db_path),
        use_lsh=True,
        lsh_params={'num_tables': 8, 'hash_size': 10, 'seed': 42}
    )

    vector_db.clear_database()

    total = embeddings.shape[0]
    for i, (emb, info) in enumerate(zip(embeddings, image_ids), start=1):
        if isinstance(info, bytes):
            info = info.decode("utf-8")
        info_str = str(info)

        path_norm = info_str.replace("\\", "/")
        parts = path_norm.split("/")
        if len(parts) >= 2:
            category = parts[-2]
            filename = parts[-1]
        else:
            category = "unknown"
            filename = path_norm

        vec_id = f"img_{i:05d}"
        metadata = {
            "filename": filename,
            "image_path": path_norm,
            "category": category,
            "source": "caltech101_embedded",
            "embedding_dim": int(emb.shape[0]),
        }

        vector_db.add_vector(vec_id, emb.astype("float32"), metadata)

        if i % 500 == 0 or i == total:
            print(f"Added {i}/{total} vectors")

    vector_db.save_to_disk()
    print("\nSaved Caltech-101 vector database to:", db_path)


if __name__ == "__main__":
    main()
