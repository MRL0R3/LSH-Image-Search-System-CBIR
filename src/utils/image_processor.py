import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import logging
from typing import List, Tuple, Optional, Union
import cv2


class ImageProcessor:


    def __init__(self, device: str = 'auto'):

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logging.info(f"Using device: {self.device}")

        self.model = self._load_pretrained_resnet()

        self.transform = self._get_transforms()

        self.model.eval()

        logging.info("ImageProcessor initialized successfully")

    def process_image(self, image_path: str) -> np.ndarray:

        embedding = self.extract_embedding(image_path)
        if embedding is None:
            raise ValueError(f"Failed to extract embedding for image: {image_path}")
        return embedding

    def _load_pretrained_resnet(self) -> nn.Module:
        model = models.resnet18(pretrained=True)

        model.fc = nn.Identity()

        model = model.to(self.device)

        return model

    def _get_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def load_image(self, image_path: str) -> Optional[Image.Image]:

        try:
            img = cv2.imread(image_path)
            if img is None:
                logging.error(f"Failed to load image: {image_path}")
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img)
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {str(e)}")
            return None

    def preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device)

    def extract_embedding(self, image: Union[str, Image.Image, np.ndarray]) -> Optional[np.ndarray]:

        try:
            if isinstance(image, str):
                img = self.load_image(image)
                if img is None:
                    return None
            else:
                img = image

            tensor = self.preprocess_image(img)

            with torch.no_grad():
                features = self.model(tensor)

            embedding = features.cpu().numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)
            return embedding.astype(np.float32)

        except Exception as e:
            logging.error(f"Error extracting embedding: {str(e)}")
            return None

    def process_directory(self, directory_path: str, max_images: int = None) -> List[Tuple[str, np.ndarray, dict]]:

        results = []
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

        image_files = [
            f for f in os.listdir(directory_path)
            if os.path.splitext(f.lower())[1] in supported_extensions
        ]

        if max_images:
            image_files = image_files[:max_images]

        logging.info(f"Processing {len(image_files)} images from {directory_path}")

        for i, filename in enumerate(image_files, 1):
            file_path = os.path.join(directory_path, filename)
            image_id = f"img_{os.path.splitext(filename)[0]}"

            embedding = self.extract_embedding(file_path)

            if embedding is not None:
                metadata = {
                    'filename': filename,
                    'path': file_path,
                    'category': os.path.basename(directory_path),
                    'original_size': Image.open(file_path).size if os.path.exists(file_path) else None,
                    'embedding_dim': len(embedding)
                }

                results.append((image_id, embedding, metadata))
                logging.info(f"Processed {i}/{len(image_files)}: {filename} -> Dim: {len(embedding)}")
            else:
                logging.warning(f"Failed to process {filename}")

        logging.info(f"Successfully processed {len(results)}/{len(image_files)} images")
        return results

    def visualize_embeddings(self, embeddings: np.ndarray, labels: List[str],
                             output_path: str = "results/plots/embeddings_tsne.png"):

        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt

            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
            embeddings_2d = tsne.fit_transform(embeddings)

            plt.figure(figsize=(12, 10))

            unique_labels = list(set(labels))
            colors = plt.cm.get_cmap('tab20', len(unique_labels))

            for i, label in enumerate(unique_labels):
                indices = [j for j, l in enumerate(labels) if l == label]
                plt.scatter(
                    embeddings_2d[indices, 0],
                    embeddings_2d[indices, 1],
                    c=[colors(i)],
                    label=label,
                    alpha=0.7,
                    s=50
                )

            plt.title('t-SNE Visualization of Image Embeddings', fontsize=16)
            plt.xlabel('t-SNE Dimension 1', fontsize=12)
            plt.ylabel('t-SNE Dimension 2', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logging.info(f"Embedding visualization saved to {output_path}")

        except Exception as e:
            logging.error(f"Error in visualization: {str(e)}")
