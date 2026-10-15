import sys
import os
import numpy as np
try:
    import numpy.exceptions
except ImportError:
    import types
    _exc = types.ModuleType('numpy.exceptions')
    for attr in dir(np):
        if 'Warning' in attr or 'Error' in attr:
            setattr(_exc, attr, getattr(np, attr))
    sys.modules['numpy.exceptions'] = _exc
from pathlib import Path
import time


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.image_processor import ImageProcessor
from src.core.vector_db import VectorDatabase

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QScrollArea, QGridLayout,
    QComboBox, QSpinBox, QProgressBar, QTabWidget, QTextEdit,
    QGroupBox, QMessageBox, QLineEdit, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap

# بقیه import ها
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import logging
import cv2


class BuildDatabaseThread(QThread):

    """Thread for building database from image folder"""

    progress = pyqtSignal(int, str)  # (percentage, message)
    finished = pyqtSignal(int, float)  # (num_images, time_taken)
    error = pyqtSignal(str)

    def __init__(self, vector_db, image_processor, folder_path, category_name):
        super().__init__()
        self.vector_db = vector_db
        self.image_processor = image_processor
        self.folder_path = folder_path
        self.category_name = category_name

    def run(self):
        try:
            start_time = time.time()
            results = self.image_processor.process_directory(
                self.folder_path,
                max_images=None
            )

            total = len(results)
            if total == 0:
                self.finished.emit(0, 0.0)
                return

            for idx, (image_id, embedding, metadata) in enumerate(results):
                metadata['category'] = self.category_name
                unique_id = f"{self.category_name}_{image_id}"
                self.vector_db.add_vector(unique_id, embedding, metadata)

                progress = int((idx + 1) / total * 100)
                self.progress.emit(progress, f"Processing {idx + 1}/{total} images...")

            elapsed = time.time() - start_time
            self.finished.emit(total, elapsed)
        except Exception as e:
            self.error.emit(str(e))


class SearchThread(QThread):
    """ Thread for asynchronous search (with HNSW support)"""
    finished = pyqtSignal(list, float, str)  # (results, time_taken, method)
    error = pyqtSignal(str)

    def __init__(self, vector_db, query_vector, top_k, search_method):
        """
        Args:
            search_method: 'lsh', 'brute-force', یا 'hnsw'
        """
        super().__init__()
        self.vector_db = vector_db
        self.query_vector = query_vector
        self.top_k = top_k
        self.search_method = search_method

    def run(self):
        try:
            start_time = time.time()

            if self.search_method == 'hnsw':
                results = self.vector_db.find_similar(
                    self.query_vector,
                    top_k=self.top_k,
                    use_lsh=False,
                    use_hnsw=True
                )
                method = "HNSW (BONUS)"
            elif self.search_method == 'lsh':
                results = self.vector_db.find_similar(
                    self.query_vector,
                    top_k=self.top_k,
                    use_lsh=True,
                    use_hnsw=False
                )
                method = "LSH"
            else:  # brute-force
                results = self.vector_db.find_similar(
                    self.query_vector,
                    top_k=self.top_k,
                    use_lsh=False,
                    use_hnsw=False
                )
                method = "Brute-force"

            elapsed = time.time() - start_time
            self.finished.emit(results, elapsed, method)
        except Exception as e:
            self.error.emit(str(e))


class EmbeddingCanvas(FigureCanvas):
    """Canvas for displaying scatter plot of embeddings"""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

    def plot_embeddings(self, vector_db, query_id=None):
        """Display 2D projection of embeddings using PCA"""
        self.axes.clear()

        if len(vector_db.vectors) == 0:
            self.axes.text(0.5, 0.5, 'No data in database',
                          ha='center', va='center', fontsize=14)
            self.draw()
            return

        vectors = []
        categories = []
        ids = []

        for vec_id, vector in vector_db.vectors.items():
            vectors.append(vector)
            metadata = vector_db.metadata.get(vec_id, {})
            category = metadata.get('category', 'unknown')
            categories.append(category)
            ids.append(vec_id)

        vectors = np.array(vectors)

        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(vectors)

        unique_categories = list(set(categories))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))
        category_colors = {cat: colors[i] for i, cat in enumerate(unique_categories)}

        for i, category in enumerate(unique_categories):
            mask = np.array(categories) == category
            self.axes.scatter(
                vectors_2d[mask, 0],
                vectors_2d[mask, 1],
                c=[category_colors[category]],
                label=category,
                alpha=0.6,
                s=50
            )

        if query_id and query_id in ids:
            query_idx = ids.index(query_id)
            self.axes.scatter(
                vectors_2d[query_idx, 0],
                vectors_2d[query_idx, 1],
                c='red',
                marker='*',
                s=500,
                edgecolors='black',
                linewidths=2,
                label='Query',
                zorder=5
            )

        self.axes.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=10)
        self.axes.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=10)
        self.axes.set_title('Embedding Space Visualization (PCA)', fontsize=12, fontweight='bold')
        self.axes.legend(loc='best', fontsize=8)
        self.axes.grid(True, alpha=0.3)
        self.draw()


class ImageLabel(QWidget):
    """Widget for displaying image with information - fixed version"""

    def __init__(self, image_path, similarity=None, vec_id=None):
        super().__init__()
        self.setStyleSheet("border: 2px solid #ccc; border-radius: 5px; padding: 5px; background-color: white;")
        self.setFixedSize(200, 250)

        # Layout اصلی عمودی
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Label برای نمایش عکس
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(180, 180)
        self.image_label.setStyleSheet("border: 1px solid #ddd; background-color: #fafafa;")

        # بارگذاری عکس
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(180, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
        else:
            # Placeholder برای عکس یافت نشده
            self.image_label.setText("Image\nNot Found")
            self.image_label.setStyleSheet(
                "border: 1px dashed #ff6b6b; background-color: #fff5f5; "
                "color: #ff6b6b; font-size: 11px;"
            )
            self.image_label.setWordWrap(True)

        layout.addWidget(self.image_label)

        # Label برای نمایش اطلاعات (vec_id و similarity)
        if similarity is not None and vec_id is not None:
            info_label = QLabel(f"<b>{vec_id}</b><br>Similarity: {similarity:.4f}")
            info_label.setAlignment(Qt.AlignCenter)
            info_label.setWordWrap(True)
            info_label.setStyleSheet("font-size: 10px; color: #555; padding: 3px;")
            layout.addWidget(info_label)

        layout.addStretch()


class CBIRMainWindow(QMainWindow):
    """Main window of image retrieval system (with bonus features)"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CBIR using LSH + HNSW (BONUS) - SBU DS&A Project")
        self.setGeometry(100, 100, 1400, 900)

        self.vector_db = None
        self.image_processor = None
        self.query_image_path = None
        self.query_embedding = None

        # مسیر دیتابیس
        self.db_path = str(PROJECT_ROOT / "data" / "embeddings" / "caltech101_db.pkl")

        self.init_ui()
        self.initialize_database()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        db_tab = QWidget()
        tabs.addTab(db_tab, " Database Management")
        self.setup_database_tab(db_tab)

        search_tab = QWidget()
        tabs.addTab(search_tab, " Image Search")
        self.setup_search_tab(search_tab)

        perf_tab = QWidget()
        tabs.addTab(perf_tab, " Performance Analysis")
        self.setup_performance_tab(perf_tab)


        bonus_tab = QWidget()
        tabs.addTab(bonus_tab, " BONUS: Complete Comparison")
        self.setup_bonus_tab(bonus_tab)

        viz_tab = QWidget()
        tabs.addTab(viz_tab, " Embedding Visualization")
        self.setup_visualization_tab(viz_tab)

        stats_tab = QWidget()
        tabs.addTab(stats_tab, " Database Statistics")
        self.setup_stats_tab(stats_tab)

        self.statusBar().showMessage("Ready")


    def find_image_path(self, stored_path):
        """
         Finding the correct image path from the stored path
        Specifically handles Caltech-101 dataset structure with category folders
        """
        if not stored_path:
            return None

        img_path = Path(stored_path)


        search_locations = [
            img_path,
            PROJECT_ROOT / stored_path,
            PROJECT_ROOT / img_path.name,
        ]


        data_folders = [
            PROJECT_ROOT / "data" / "raw_images",
            PROJECT_ROOT / "data" / "raw_images" / "sample_synthetic",
            PROJECT_ROOT / "data" / "caltech101" / "images" / "101_ObjectCategories",
        ]

        for folder in data_folders:
            # search with name
            search_locations.append(folder / img_path.name)

            # search in subfolder
            if '/' in stored_path or '\\' in stored_path:
                parts = Path(stored_path).parts
                for i in range(len(parts)):
                    partial_path = Path(*parts[i:])
                    search_locations.append(folder / partial_path)

        # find the first valid subfolder
        for path in search_locations:
            try:
                if path.exists() and path.is_file():
                    return path
            except:
                continue

        return None

    def setup_database_tab(self, parent):
        layout = QVBoxLayout(parent)

        sample_group = QGroupBox("Build Synthetic Sample Dataset (Automatic)")
        sample_layout = QVBoxLayout()

        self.sample_info_label = QLabel(
            "This will generate a synthetic dataset with categories: "
            "car, animal, building, food, nature\n"
            "Then it creates embeddings and fills the vector database."
        )
        self.sample_info_label.setWordWrap(True)
        sample_layout.addWidget(self.sample_info_label)

        self.build_sample_btn = QPushButton("Build Sample Dataset & Database")
        self.build_sample_btn.setStyleSheet(
            "font-size: 14px; padding: 10px; background-color: #3F51B5; color: white;"
        )
        self.build_sample_btn.clicked.connect(self.build_sample_dataset_and_database)
        sample_layout.addWidget(self.build_sample_btn)

        self.sample_progress = QProgressBar()
        self.sample_progress.setVisible(False)
        sample_layout.addWidget(self.sample_progress)

        sample_group.setLayout(sample_layout)
        layout.addWidget(sample_group)

        create_group = QGroupBox("Build Database from Images")
        create_layout = QVBoxLayout()

        folder_layout = QHBoxLayout()
        self.folder_path_label = QLabel("No folder selected")
        self.folder_path_label.setStyleSheet("padding: 5px; border: 1px solid #ccc;")
        folder_layout.addWidget(QLabel("Image Folder:"))
        folder_layout.addWidget(self.folder_path_label, 1)

        self.select_folder_btn = QPushButton("Select Folder")
        self.select_folder_btn.clicked.connect(self.select_image_folder)
        folder_layout.addWidget(self.select_folder_btn)
        create_layout.addLayout(folder_layout)

        category_layout = QHBoxLayout()
        category_layout.addWidget(QLabel("Category Name:"))
        self.category_input = QLineEdit()
        self.category_input.setPlaceholderText("e.g., animals, cars, buildings")
        category_layout.addWidget(self.category_input)
        create_layout.addLayout(category_layout)

        self.build_db_btn = QPushButton("Add Images to Database")
        self.build_db_btn.setStyleSheet(
            "font-size: 14px; padding: 10px; background-color: #4CAF50; color: white;"
        )
        self.build_db_btn.clicked.connect(self.build_database)
        create_layout.addWidget(self.build_db_btn)

        self.build_progress = QProgressBar()
        self.build_progress.setVisible(False)
        create_layout.addWidget(self.build_progress)

        create_group.setLayout(create_layout)
        layout.addWidget(create_group)

        manage_group = QGroupBox("Database Operations")
        manage_layout = QHBoxLayout()

        self.clear_db_btn = QPushButton("Clear Database")
        self.clear_db_btn.clicked.connect(self.clear_database)
        manage_layout.addWidget(self.clear_db_btn)

        self.save_db_btn = QPushButton("Save Database")
        self.save_db_btn.clicked.connect(self.save_database)
        manage_layout.addWidget(self.save_db_btn)

        self.load_db_btn = QPushButton("Load Database")
        self.load_db_btn.clicked.connect(self.load_database)
        manage_layout.addWidget(self.load_db_btn)

        manage_group.setLayout(manage_layout)
        layout.addWidget(manage_group)

        log_group = QGroupBox("Build Log")
        log_layout = QVBoxLayout()
        self.db_log = QTextEdit()
        self.db_log.setReadOnly(True)
        self.db_log.setMaximumHeight(200)
        log_layout.addWidget(self.db_log)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        layout.addStretch()

    def generate_synthetic_image(self, category):

        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


        color_map = {
            'car': (200, 50, 50),
            'animal': (50, 200, 50),
            'building': (100, 100, 200),
            'food': (200, 200, 50),
            'nature': (50, 150, 100)
        }

        color = color_map.get(category, (128, 128, 128))
        cv2.rectangle(img, (50, 50), (174, 174), color, -1)
        cv2.putText(img, category, (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return img

    def build_sample_dataset_and_database(self):

        reply = QMessageBox.question(
            self,
            "Confirm",
            "This will CLEAR current database and build a new synthetic dataset.\nContinue?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        try:
            self.statusBar().showMessage("Building synthetic dataset and database...")
            self.sample_progress.setVisible(True)
            self.sample_progress.setValue(0)

            synthetic_db_path = str(PROJECT_ROOT / "data" / "embeddings" / "lsh_test_db.pkl")
            self.db_path = synthetic_db_path
            self.vector_db.persist_path = synthetic_db_path
            self.vector_db.clear_database()

            categories = ['car', 'animal', 'building', 'food', 'nature']
            num_per_cat = 20
            total_images = len(categories) * num_per_cat
            processed = 0

            sample_root = PROJECT_ROOT / "data" / "raw_images" / "sample_synthetic"
            sample_root.mkdir(parents=True, exist_ok=True)

            start_time = time.time()

            for cat in categories:
                cat_dir = sample_root / cat
                cat_dir.mkdir(parents=True, exist_ok=True)

                for i in range(num_per_cat):
                    img = self.generate_synthetic_image(cat)
                    filename = f"{cat}_{i+1:03d}.jpg"
                    img_path = cat_dir / filename

                    cv2.imwrite(str(img_path), img)


                    embedding = self.image_processor.process_image(str(img_path))

                    metadata = {
                        "image_path": str(img_path.resolve()),
                        "relative_path": str(img_path.relative_to(PROJECT_ROOT)),
                        "category": cat,
                        "filename": filename
                    }

                    vec_id = f"{cat}_{i+1:03d}"
                    self.vector_db.add_vector(vec_id, embedding, metadata)

                    processed += 1
                    progress = int(processed / total_images * 100)
                    self.sample_progress.setValue(progress)
                    self.statusBar().showMessage(
                        f"Building synthetic dataset: {processed}/{total_images} images..."
                    )

            elapsed = time.time() - start_time
            self.vector_db.save_to_disk()

            self.sample_progress.setVisible(False)
            self.db_log.append(
                f"\n[OK] Synthetic dataset built with {total_images} images in {elapsed:.2f} seconds"
            )
            self.db_log.append(f"Database saved to {self.db_path}\n")

            self.statusBar().showMessage(
                f"Synthetic database ready: {len(self.vector_db.vectors)} images"
            )

            self.update_stats()
            self.refresh_embedding_plot()

            QMessageBox.information(
                self,
                "Success",
                f"Synthetic dataset & database built!\nTotal images: {len(self.vector_db.vectors)}"
            )
        except Exception as e:
            self.sample_progress.setVisible(False)
            QMessageBox.critical(self, "Error", f"Failed to build synthetic dataset:\n{str(e)}")
            logging.error(f"Synthetic dataset build error: {e}")

    def select_image_folder(self):

        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.folder_path_label.setText(folder)
            self.selected_folder = folder

    def build_database(self):

        if not hasattr(self, 'selected_folder'):
            QMessageBox.warning(self, "Warning", "Please select an image folder first!")
            return

        category = self.category_input.text().strip()
        if not category:
            QMessageBox.warning(self, "Warning", "Please enter a category name!")
            return

        self.build_db_btn.setEnabled(False)
        self.build_progress.setVisible(True)
        self.build_progress.setValue(0)

        self.db_log.append(f"\n{'=' * 50}")
        self.db_log.append(f"Building database from: {self.selected_folder}")
        self.db_log.append(f"Category: {category}")
        self.db_log.append(f"{'=' * 50}\n")

        # ساخت thread
        self.build_thread = BuildDatabaseThread(
            self.vector_db,
            self.image_processor,
            self.selected_folder,
            category
        )

        self.build_thread.progress.connect(self.on_build_progress)
        self.build_thread.finished.connect(self.on_build_finished)
        self.build_thread.error.connect(self.on_build_error)
        self.build_thread.start()

    def on_build_progress(self, percentage, message):
        self.build_progress.setValue(percentage)
        self.statusBar().showMessage(message)

    def on_build_finished(self, num_images, elapsed):
        self.build_progress.setVisible(False)
        self.build_db_btn.setEnabled(True)

        self.db_log.append(f" Added {num_images} images in {elapsed:.2f} seconds")
        self.db_log.append(f" Auto-saved to disk\n")

        self.statusBar().showMessage(f"Database updated: {len(self.vector_db.vectors)} total images")
        self.update_stats()

    def on_build_error(self, error_msg):
        self.build_progress.setVisible(False)
        self.build_db_btn.setEnabled(True)
        self.db_log.append(f" Error: {error_msg}\n")
        QMessageBox.critical(self, "Error", f"Failed to build database:\n{error_msg}")

    def clear_database(self):

        reply = QMessageBox.question(
            self,
            "Confirm",
            "Are you sure you want to clear the entire database?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.vector_db.clear_database()
            self.db_log.append("Database cleared\n")
            self.statusBar().showMessage("Database cleared")
            self.update_stats()

    def save_database(self):

        try:
            self.vector_db.save_to_disk()
            self.db_log.append(f"Database saved to {self.db_path}\n")
            QMessageBox.information(self, "Success", "Database saved successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save database:\n{str(e)}")

    def load_database(self):
        """بارگذاری دیتابیس"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Database",
            str(PROJECT_ROOT / "data" / "embeddings"),
            "Pickle Files (*.pkl)"
        )

        if file_path:
            try:
                self.vector_db = VectorDatabase(
                    dim=512,
                    persist_path=file_path,
                    use_lsh=True,
                    lsh_params={'num_tables': 8, 'hash_size': 10, 'seed': 42}
                )

                self.db_path = file_path
                self.db_log.append(f" Loaded database from {file_path}\n")
                self.db_log.append(f"   Total images: {len(self.vector_db.vectors)}\n")

                self.update_stats()
                self.refresh_embedding_plot()

                QMessageBox.information(
                    self,
                    "Success",
                    f"Database loaded successfully!\nTotal images: {len(self.vector_db.vectors)}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load database:\n{str(e)}")

    def setup_search_tab(self, parent):
        layout = QVBoxLayout(parent)

        control_layout = QHBoxLayout()

        self.select_btn = QPushButton(" Select Query Image")
        self.select_btn.clicked.connect(self.select_query_image)
        self.select_btn.setStyleSheet("font-size: 14px; padding: 10px;")
        control_layout.addWidget(self.select_btn)

        self.top_k_label = QLabel("Top K:")
        control_layout.addWidget(self.top_k_label)

        self.top_k_spinbox = QSpinBox()
        self.top_k_spinbox.setRange(1, 50)
        self.top_k_spinbox.setValue(10)
        control_layout.addWidget(self.top_k_spinbox)

        self.search_method_combo = QComboBox()
        self.search_method_combo.addItems(["LSH (Fast)", "Brute-force (Exact)"])
        control_layout.addWidget(self.search_method_combo)

        self.search_btn = QPushButton(" Search Similar Images")
        self.search_btn.clicked.connect(self.search_similar_images)
        self.search_btn.setEnabled(False)
        self.search_btn.setStyleSheet(
            "font-size: 14px; padding: 10px; background-color: #4CAF50; color: white;"
        )
        control_layout.addWidget(self.search_btn)
        control_layout.addStretch()

        layout.addLayout(control_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.search_time_label = QLabel("")
        self.search_time_label.setStyleSheet("font-size: 12px; color: #0066cc; padding: 5px;")
        layout.addWidget(self.search_time_label)

        query_group_layout = QHBoxLayout()
        self.query_image_label = QLabel()
        self.query_image_label.setFixedSize(300, 300)
        self.query_image_label.setAlignment(Qt.AlignCenter)
        self.query_image_label.setStyleSheet(
            "border: 3px solid #2196F3; border-radius: 10px; background-color: #f0f0f0;"
        )
        self.query_image_label.setText("No query image selected")
        query_group_layout.addWidget(self.query_image_label)
        query_group_layout.addStretch()
        layout.addLayout(query_group_layout)

        results_label = QLabel("Search Results:")
        results_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(results_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: white;")
        self.results_widget = QWidget()
        self.results_layout = QGridLayout(self.results_widget)
        self.results_layout.setSpacing(15)
        scroll.setWidget(self.results_widget)
        layout.addWidget(scroll)

    def setup_performance_tab(self, parent):
        layout = QVBoxLayout(parent)

        compare_btn = QPushButton("Run Performance Comparison (LSH vs Brute-force)")
        compare_btn.setStyleSheet("font-size: 14px; padding: 10px; background-color: #FF9800; color: white;")
        compare_btn.clicked.connect(self.run_performance_comparison)
        layout.addWidget(compare_btn)

        self.perf_table = QTableWidget()
        self.perf_table.setColumnCount(4)
        self.perf_table.setHorizontalHeaderLabels(["Method", "Time (ms)", "Results Found", "Speed Ratio"])
        layout.addWidget(self.perf_table)

        self.perf_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        layout.addWidget(self.perf_canvas)

    def setup_bonus_tab(self, parent):
        """ BONUS: Complete comparison tab for all algorithms"""
        layout = QVBoxLayout(parent)

        compare_btn = QPushButton(" Compare ALL Algorithms (Brute-force + LSH + HNSW)")
        compare_btn.setStyleSheet(
            "font-size: 14px; padding: 12px; background-color: #9C27B0; color: white; font-weight: bold;"
        )
        compare_btn.clicked.connect(self.run_complete_comparison)
        layout.addWidget(compare_btn)

        info_label = QLabel(
            "This feature compares all available search methods:\n"
            "• Brute-force (Ground Truth)\n"
            "• LSH (Required)\n"
            "• HNSW (BONUS) - if available\n\n"
            "Select a query image first!"
        )
        info_label.setStyleSheet("font-size: 11px; padding: 10px; background-color: #FFF3E0; border-radius: 5px;")
        layout.addWidget(info_label)

        self.bonus_table = QTableWidget()
        self.bonus_table.setColumnCount(5)
        self.bonus_table.setHorizontalHeaderLabels([
            "Algorithm", "Time (ms)", "Speedup", "Precision@10", "Status"
        ])
        self.bonus_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.bonus_table)

        self.bonus_canvas = FigureCanvas(Figure(figsize=(12, 6)))
        layout.addWidget(self.bonus_canvas)

    def setup_visualization_tab(self, parent):
        layout = QVBoxLayout(parent)

        control_layout = QHBoxLayout()
        refresh_btn = QPushButton("Refresh Plot")
        refresh_btn.clicked.connect(self.refresh_embedding_plot)
        refresh_btn.setStyleSheet("padding: 8px; font-size: 13px;")
        control_layout.addWidget(refresh_btn)
        control_layout.addStretch()
        layout.addLayout(control_layout)

        self.embedding_canvas = EmbeddingCanvas(self, width=10, height=7)
        layout.addWidget(self.embedding_canvas)

    def setup_stats_tab(self, parent):
        layout = QVBoxLayout(parent)

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setStyleSheet("font-family: 'Courier New'; font-size: 12px;")
        layout.addWidget(self.stats_text)

        refresh_stats_btn = QPushButton("Refresh Statistics")
        refresh_stats_btn.clicked.connect(self.update_stats)
        refresh_stats_btn.setStyleSheet("padding: 10px; font-size: 13px;")
        layout.addWidget(refresh_stats_btn)

    def _update_search_methods_combo(self):
        """ Update search options based on database capabilities"""
        current_selection = self.search_method_combo.currentIndex()
        self.search_method_combo.clear()

        methods = ["LSH (Fast)", "Brute-force (Exact)"]


        if self.vector_db and self.vector_db.use_hnsw:
            methods.append(" HNSW (BONUS)")
            logging.info(" HNSW search method added to UI")

        self.search_method_combo.addItems(methods)

        if current_selection < len(methods):
            self.search_method_combo.setCurrentIndex(current_selection)
        else:
            self.search_method_combo.setCurrentIndex(0)

    def initialize_database(self):
        """Load database (with HNSW support)"""
        try:
            self.statusBar().showMessage(f"Initializing database from {self.db_path}...")
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)


            self.vector_db = VectorDatabase(
                dim=512,
                persist_path=self.db_path,
                use_lsh=True,
                use_hnsw=True,
                lsh_params={'num_tables': 8, 'hash_size': 10, 'seed': 42},
                hnsw_params={'M': 16, 'ef_construction': 200}
            )

            self.image_processor = ImageProcessor(device='auto')

            # update ComboBox
            self._update_search_methods_combo()

            num_images = len(self.vector_db.vectors)
            hnsw_status = " HNSW Enabled" if self.vector_db.use_hnsw else "️ HNSW Disabled"

            self.statusBar().showMessage(
                f"Database loaded: {num_images} images | {hnsw_status}"
            )

            if num_images > 0:
                self.update_stats()
                self.refresh_embedding_plot()
        except Exception as e:
            self.statusBar().showMessage(f"Error initializing database: {str(e)}")
            logging.error(f"Database initialization error: {e}")

    def select_query_image(self):
        """انتخاب تصویر query"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Query Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            self.query_image_path = file_path
            pixmap = QPixmap(file_path)
            pixmap = pixmap.scaled(280, 280, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.query_image_label.setPixmap(pixmap)

            try:
                self.query_embedding = self.image_processor.process_image(file_path)
                self.search_btn.setEnabled(True)
                self.statusBar().showMessage(f"Query image loaded: {Path(file_path).name}")
            except Exception as e:
                self.statusBar().showMessage(f"Error processing image: {str(e)}")
                self.search_btn.setEnabled(False)

    def search_similar_images(self):
        """Search similar images (with HNSW support)"""
        if self.query_embedding is None:
            return

        if len(self.vector_db.vectors) == 0:
            QMessageBox.warning(self, "Warning", "Database is empty!")
            return


        for i in reversed(range(self.results_layout.count())):
            self.results_layout.itemAt(i).widget().setParent(None)

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.search_btn.setEnabled(False)

        top_k = self.top_k_spinbox.value()

        # Determine search method
        method_text = self.search_method_combo.currentText()
        if "LSH" in method_text:
            search_method = 'lsh'
        elif "Brute-force" in method_text:
            search_method = 'brute-force'
        else:  # HNSW
            search_method = 'hnsw'

        self.search_thread = SearchThread(
            self.vector_db,
            self.query_embedding,
            top_k,
            search_method
        )

        self.search_thread.finished.connect(self.display_results)
        self.search_thread.error.connect(self.search_error)
        self.search_thread.start()

    def display_results(self, results, elapsed, method):
        """ Display search results with correct image loading"""
        self.progress_bar.setVisible(False)
        self.search_btn.setEnabled(True)

        self.search_time_label.setText(
            f" Search completed using {method} in {elapsed * 1000:.2f} ms | Found {len(results)} images"
        )

        if not results:
            no_result_label = QLabel("No results found")
            no_result_label.setStyleSheet("font-size: 16px; color: red;")
            self.results_layout.addWidget(no_result_label, 0, 0)
            return

        cols = 5
        for idx, (vec_id, similarity) in enumerate(results):
            row = idx // cols
            col = idx % cols

            metadata = self.vector_db.get_metadata(vec_id)
            stored_path = metadata.get('image_path', '')

            #  Find valid image path
            valid_path = self.find_image_path(stored_path)

            # If not found, use stored path (ImageLabel shows placeholder if image fails to load)
            img_path_to_use = str(valid_path) if valid_path else stored_path

            img_label = ImageLabel(img_path_to_use, similarity, vec_id)
            self.results_layout.addWidget(img_label, row, col)

        self.statusBar().showMessage(f"Found {len(results)} similar images using {method}")


    def search_error(self, error_msg):
        """Error handling in search"""
        self.progress_bar.setVisible(False)
        self.search_btn.setEnabled(True)
        self.statusBar().showMessage(f"Search error: {error_msg}")
        QMessageBox.critical(self, "Search Error", f"An error occurred during search:\n{error_msg}")

    def run_performance_comparison(self):
        """Performance comparison of LSH and Brute-force"""
        if len(self.vector_db.vectors) == 0:
            QMessageBox.warning(self, "Warning", "Database is empty!")
            return

        if self.query_embedding is None:
            QMessageBox.warning(self, "Warning", "Please select a query image first!")
            return

        try:
            self.statusBar().showMessage("Running performance comparison...")

            start = time.time()
            bf_results = self.vector_db.find_similar(self.query_embedding, top_k=10, use_lsh=False)
            bf_time = (time.time() - start) * 1000

            start = time.time()
            lsh_results = self.vector_db.find_similar(self.query_embedding, top_k=10, use_lsh=True)
            lsh_time = (time.time() - start) * 1000

            speed_ratio = bf_time / lsh_time if lsh_time > 0 else 0

            self.perf_table.setRowCount(2)
            self.perf_table.setItem(0, 0, QTableWidgetItem("Brute-force (Exact)"))
            self.perf_table.setItem(0, 1, QTableWidgetItem(f"{bf_time:.2f}"))
            self.perf_table.setItem(0, 2, QTableWidgetItem(str(len(bf_results))))
            self.perf_table.setItem(0, 3, QTableWidgetItem("1.0x (baseline)"))

            self.perf_table.setItem(1, 0, QTableWidgetItem("LSH (Approximate)"))
            self.perf_table.setItem(1, 1, QTableWidgetItem(f"{lsh_time:.2f}"))
            self.perf_table.setItem(1, 2, QTableWidgetItem(str(len(lsh_results))))
            self.perf_table.setItem(1, 3, QTableWidgetItem(f"{speed_ratio:.2f}x faster"))

            ax = self.perf_canvas.figure.clear()
            ax = self.perf_canvas.figure.add_subplot(111)

            methods = ['Brute-force', 'LSH']
            times = [bf_time, lsh_time]
            colors = ['#FF6B6B', '#4ECDC4']

            bars = ax.bar(methods, times, color=colors, alpha=0.7)
            ax.set_ylabel('Time (ms)', fontsize=12)
            ax.set_title('Performance Comparison: Brute-force vs LSH', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            for bar, time_val in zip(bars, times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{time_val:.2f} ms',
                       ha='center', va='bottom', fontsize=10)

            self.perf_canvas.draw()

            self.statusBar().showMessage(
                f"Comparison complete: LSH is {speed_ratio:.2f}x faster than Brute-force"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Performance comparison failed:\n{str(e)}")

    def run_complete_comparison(self):

        """BONUS: Complete comparison of all algorithms"""

        if self.query_embedding is None:
            QMessageBox.warning(self, "Warning", "Please select a query image first!")
            return

        if len(self.vector_db.vectors) == 0:
            QMessageBox.warning(self, "Warning", "Database is empty!")
            return

        try:
            self.statusBar().showMessage("Running complete comparison (including BONUS)...")
            results_data = []

            # 1. Brute-force
            start = time.time()
            bf_results = self.vector_db.find_similar(
                self.query_embedding, top_k=10, use_lsh=False, use_hnsw=False
            )
            bf_time = (time.time() - start) * 1000
            bf_ids = set(vid for vid, _ in bf_results)

            results_data.append({
                'name': 'Brute-force',
                'time': bf_time,
                'results': bf_results,
                'color': '#FF6B6B'
            })

            # 2. LSH
            start = time.time()
            lsh_results = self.vector_db.find_similar(
                self.query_embedding, top_k=10, use_lsh=True, use_hnsw=False
            )
            lsh_time = (time.time() - start) * 1000
            lsh_ids = set(vid for vid, _ in lsh_results)
            lsh_precision = len(lsh_ids & bf_ids) / 10 * 100 if lsh_results else 0

            results_data.append({
                'name': 'LSH',
                'time': lsh_time,
                'results': lsh_results,
                'precision': lsh_precision,
                'speedup': bf_time / lsh_time if lsh_time > 0 else 0,
                'color': '#4ECDC4'
            })

            # 3. HNSW (if enabled)
            if self.vector_db.use_hnsw and self.vector_db.hnsw_index:
                start = time.time()
                hnsw_results = self.vector_db.find_similar(
                    self.query_embedding, top_k=10, use_hnsw=True
                )
                hnsw_time = (time.time() - start) * 1000
                hnsw_ids = set(vid for vid, _ in hnsw_results)
                hnsw_precision = len(hnsw_ids & bf_ids) / 10 * 100 if hnsw_results else 0

                results_data.append({
                    'name': 'HNSW (BONUS)',
                    'time': hnsw_time,
                    'results': hnsw_results,
                    'precision': hnsw_precision,
                    'speedup': bf_time / hnsw_time if hnsw_time > 0 else 0,
                    'color': '#9C27B0'
                })

            # fill table
            self.bonus_table.setRowCount(len(results_data))
            for i, data in enumerate(results_data):
                self.bonus_table.setItem(i, 0, QTableWidgetItem(data['name']))
                self.bonus_table.setItem(i, 1, QTableWidgetItem(f"{data['time']:.2f}"))

                speedup = data.get('speedup', 1.0)
                self.bonus_table.setItem(i, 2, QTableWidgetItem(f"{speedup:.2f}x"))

                precision = data.get('precision', 100.0)
                self.bonus_table.setItem(i, 3, QTableWidgetItem(f"{precision:.1f}%"))

                status = " Ground Truth" if i == 0 else f" {precision:.0f}% Accurate"
                self.bonus_table.setItem(i, 4, QTableWidgetItem(status))

#Plot chart
            self._plot_complete_comparison(results_data)

            self.statusBar().showMessage(" Complete comparison finished!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Complete comparison failed:\n{str(e)}")
            logging.error(f"Comparison error: {e}")

    def _plot_complete_comparison(self, results_data):
        """Plot complete comparison chart"""
        fig = self.bonus_canvas.figure
        fig.clear()

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        names = [d['name'] for d in results_data]
        times = [d['time'] for d in results_data]
        colors = [d['color'] for d in results_data]

        # tim chart
        bars1 = ax1.bar(names, times, color=colors, alpha=0.7)
        ax1.set_ylabel('Time (ms)', fontsize=11)
        ax1.set_title('Search Time Comparison', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        for bar, time_val in zip(bars1, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{time_val:.1f}ms',
                    ha='center', va='bottom', fontsize=9)

        # acc chart
        precisions = [d.get('precision', 100.0) for d in results_data]
        bars2 = ax2.bar(names, precisions, color=colors, alpha=0.7)
        ax2.set_ylabel('Precision@10 (%)', fontsize=11)
        ax2.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 105)
        ax2.grid(axis='y', alpha=0.3)

        for bar, prec in zip(bars2, precisions):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{prec:.1f}%',
                    ha='center', va='bottom', fontsize=9)

        fig.tight_layout()
        self.bonus_canvas.draw()

    def refresh_embedding_plot(self):
        """Update embedding plot"""

        if self.vector_db and len(self.vector_db.vectors) > 0:
            try:
                self.embedding_canvas.plot_embeddings(self.vector_db)
                self.statusBar().showMessage("Embedding plot updated")
            except Exception as e:
                self.statusBar().showMessage(f"Error plotting embeddings: {str(e)}")

    def update_stats(self):
        """update stats"""
        if not self.vector_db:
            return

        stats = self.vector_db.get_database_stats()

        stats_text = "=" * 60 + "\n"
        stats_text += "DATABASE STATISTICS\n"
        stats_text += "=" * 60 + "\n\n"

        stats_text += f"Total Vectors: {stats['total_vectors']}\n"
        stats_text += f"Dimension: {stats['dimension']}\n"
        stats_text += f"LSH Enabled: {stats['use_lsh']}\n"
        stats_text += f"HNSW Enabled: {stats.get('use_hnsw', False)} {' (BONUS)' if stats.get('use_hnsw') else ''}\n"
        stats_text += f"Thread-Safe: {stats.get('thread_safe', False)} {' (BONUS)' if stats.get('thread_safe') else ''}\n"
        stats_text += f"Storage Path: {stats['persist_path']}\n\n"

        # LSH Stats
        if 'lsh_stats' in stats:
            stats_text += "-" * 60 + "\n"
            stats_text += "LSH INDEX STATISTICS\n"
            stats_text += "-" * 60 + "\n"
            lsh_stats = stats['lsh_stats']
            for key, value in lsh_stats.items():
                stats_text += f"  {key}: {value}\n"
            stats_text += "\n"

        # HNSW Stats
        if 'hnsw_stats' in stats:
            stats_text += "-" * 60 + "\n"
            stats_text += " HNSW INDEX STATISTICS (BONUS)\n"
            stats_text += "-" * 60 + "\n"
            hnsw_stats = stats['hnsw_stats']
            for key, value in hnsw_stats.items():
                stats_text += f"  {key}: {value}\n"
            stats_text += "\n"

        # Category Distribution
        categories = {}
        for vec_id in self.vector_db.get_all_ids():
            metadata = self.vector_db.get_metadata(vec_id)
            cat = metadata.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1

        if categories:
            stats_text += "-" * 60 + "\n"
            stats_text += "CATEGORY DISTRIBUTION\n"
            stats_text += "-" * 60 + "\n"
            for cat, count in sorted(categories.items()):
                stats_text += f"  {cat}: {count} images\n"

        self.stats_text.setText(stats_text)


def main():

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = CBIRMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()