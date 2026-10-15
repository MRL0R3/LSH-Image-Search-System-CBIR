import sys
import logging
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = SRC_DIR / "data"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def setup_logging():
    log_file = PROJECT_ROOT / "debug.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler()
        ]
    )


def launch_gui():
    print("=" * 60)
    print(" Launching Graphical Interface...")
    print("=" * 60)

    try:
        from gui.main_window import main as gui_main
        gui_main()
    except ImportError as e:
        print(f" Error importing GUI: {e}")
        print("Please ensure PyQt5 is installed:")
        print("  pip install PyQt5")
        sys.exit(1)
    except Exception as e:
        print(f" Error executing GUI: {e}")
        logging.error(f"GUI launch error: {e}")
        sys.exit(1)


def run_tests():
    print("=" * 60)
    print("Running LSH tests and database construction")
    print("=" * 60)

    try:
        import numpy as np
        import cv2
        from tqdm import tqdm
        from core.vector_db import VectorDatabase
        from utils.image_processor import ImageProcessor

        EMBEDDING_DIM = 512
        SAMPLE_DATA_DIR = str(DATA_DIR / "raw_images" / "sample_dataset")
        DB_PATH = str(DATA_DIR / "embeddings" / "lsh_test_db.pkl")

        generate_sample_dataset(SAMPLE_DATA_DIR, num_images_per_category=15)

        vector_db = VectorDatabase(
            dim=EMBEDDING_DIM,
            persist_path=DB_PATH,
            use_lsh=True,
            lsh_params={
                'num_tables': 8,
                'hash_size': 10,
                'seed': 42
            }
        )

        vector_db.clear_database()

        image_processor = ImageProcessor(device='auto')

        print("\n" + "=" * 40)
        print("  Processing images and populating database")
        print("=" * 40)

        categories = ['car', 'animal', 'building', 'food', 'nature']
        total_vectors = 0

        for category in tqdm(categories, desc="Processing categories"):
            category_dir = Path(SAMPLE_DATA_DIR) / category
            if not category_dir.exists():
                continue

            results = image_processor.process_directory(str(category_dir), max_images=10)

            for image_id, embedding, metadata in tqdm(results, desc=f"Inserting {category}", leave=False):
                unique_id = f"{category}_{image_id}"
                vector_db.add_vector(unique_id, embedding, metadata)
                total_vectors += 1

        print(f"\n Database populated with {total_vectors} vectors")

        test_search(vector_db)

        vector_db.save_to_disk()
        print(f"\n Database saved to {DB_PATH}")

        print("\n" + "=" * 60)
        print(" LSH test completed successfully!")
        print("=" * 60)

    except ImportError as e:
        print(f" Error: {e}")
        print("Please install required libraries:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f" Error executing test: {e}")
        logging.error(f"Test error: {e}")
        sys.exit(1)


def generate_sample_dataset(output_dir: str, num_images_per_category: int = 10):
    import numpy as np
    import cv2
    import os

    categories = ['car', 'animal', 'building', 'food', 'nature']
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating sample images in {output_dir}...")

    for category in categories:
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        for i in range(num_images_per_category):
            if category == 'car':
                img = np.ones((224, 224, 3), dtype=np.uint8) * 200
                cv2.rectangle(img, (50, 100), (174, 150), (50, 50, 200), -1)
                cv2.circle(img, (80, 160), 20, (0, 0, 0), -1)
                cv2.circle(img, (144, 160), 20, (0, 0, 0), -1)

            elif category == 'animal':
                img = np.zeros((224, 224, 3), dtype=np.uint8)
                for _ in range(50):
                    x, y = np.random.randint(0, 224, 2)
                    cv2.circle(img, (x, y), np.random.randint(5, 15),
                               (np.random.randint(100, 256),
                                np.random.randint(100, 256),
                                np.random.randint(100, 256)), -1)

            elif category == 'building':
                img = np.ones((224, 224, 3), dtype=np.uint8) * 220
                for x in range(30, 200, 40):
                    cv2.rectangle(img, (x, 50), (x + 20, 200), (100, 100, 100), -1)
                cv2.rectangle(img, (0, 180), (224, 224), (50, 150, 50), -1)

            elif category == 'food':
                img = np.ones((224, 224, 3), dtype=np.uint8) * 240
                colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 255)]
                for j in range(4):
                    x, y = 60 + j * 40, 100
                    cv2.circle(img, (x, y), 30, colors[j], -1)

            else:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
                for y in range(224):
                    color = (0, int(255 * y / 224), int(150 * (1 - y / 224)))
                    cv2.line(img, (0, y), (224, y), color, 1)


            filename = f"{category}_{i + 1:03d}.jpg"
            cv2.imwrite(os.path.join(category_dir, filename), img)

    print(f"Sample dataset generated with {len(categories) * num_images_per_category} images")


def test_search(vector_db):
    print("\n" + "=" * 40)
    print("Testing search with different methods")
    print("=" * 40)

    sample_queries = []
    test_categories = ['car', 'animal', 'building']

    for category in test_categories:
        for vec_id in vector_db.get_all_ids():
            if vec_id.startswith(category):
                vector = vector_db.get_vector(vec_id)
                if vector is not None:
                    sample_queries.append((vec_id, vector))
                    print(f"  * Sample query from category '{category}': {vec_id}")
                    break

    print("\n" + "-" * 30)
    print("Exhaustive Search (Brute-force):")
    print("-" * 30)

    for query_id, query_vec in sample_queries[:2]:
        results = vector_db.find_similar(query_vec, top_k=5, use_lsh=False)
        print(f"\nQuery: {query_id}")
        for rank, (res_id, sim) in enumerate(results, 1):
            meta = vector_db.get_metadata(res_id)
            category = meta.get('category', 'unknown')
            print(f"  #{rank}: {res_id} | Similarity: {sim:.4f} | Category: {category}")

    print("\n" + "-" * 30)
    print("LSH Search:")
    print("-" * 30)

    for query_id, query_vec in sample_queries[:2]:
        results = vector_db.find_similar(query_vec, top_k=5, use_lsh=True, lsh_max_candidates=20)
        print(f"\nQuery: {query_id}")
        for rank, (res_id, sim) in enumerate(results, 1):
            meta = vector_db.get_metadata(res_id)
            category = meta.get('category', 'unknown')
            print(f"  #{rank}: {res_id} | Similarity: {sim:.4f} | Category: {category}")


def show_menu():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "LSH Image Search System" + " " * 20 + "║")
    print("║" + " " * 10 + "Content-Based Image Retrieval System" + " " * 12 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    print("Please select:")
    print("  [1]  Launch Graphical Interface (GUI)")
    print("  [2]  Test and Build Database")
    print("  [3]  Exit")
    print()
    print("-" * 60)


def main():
    setup_logging()

    if len(sys.argv) > 1:
        if sys.argv[1] in ['--gui', '-g']:
            launch_gui()
            return
        elif sys.argv[1] in ['--test', '-t']:
            run_tests()
            return
        elif sys.argv[1] in ['--help', '-h']:
            print("\nUsage:")
            print("  python main.py          # Show menu")
            print("  python main.py --gui    # Launch GUI directly")
            print("  python main.py --test   # Run tests directly")
            print("  python main.py --help   # Show help")
            return

    while True:
        show_menu()
        try:
            choice = input("Your selection (1-3): ").strip()

            if choice == '1':
                launch_gui()
                break
            elif choice == '2':
                run_tests()
                break
            elif choice == '3':
                print("\n Exiting program...")
                sys.exit(0)
            else:
                print("\n Invalid selection! Please choose option 1, 2, or 3.")
                input("Press Enter to continue...")

        except KeyboardInterrupt:
            print("\n\n Exiting program...")
            sys.exit(0)
        except Exception as e:
            print(f"\n Error: {e}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()