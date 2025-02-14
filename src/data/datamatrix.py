import os
import io
from dataclasses import dataclass
from typing import Tuple, List, Optional
import random

from PIL import Image, ImageDraw
from pylibdmtx.pylibdmtx import encode
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import torch
from torch.utils.data import Dataset


@dataclass
class DataMatrixConfig:
    min_size: int = 100
    max_size: int = 100
    margin: int = 10
    data_length: int = 8


class DataMatrixGenerator:
    def __init__(self, config: DataMatrixConfig):
        self.config = config

    def generate_random_data(self) -> str:
        """Generate random data for the DataMatrix."""
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return "".join(random.choices(chars, k=self.config.data_length))

    def create_datamatrix(self, data: Optional[str] = None) -> Image.Image:
        """Create a DataMatrix image with the given data."""
        if data is None:
            data = self.generate_random_data()
        encoded = encode(data.encode("utf8"))
        return Image.frombytes("RGB", (encoded.width, encoded.height), encoded.pixels)

    def get_random_position(
        self, image_size: Tuple[int, int], datamatrix_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Get a random position for the DataMatrix."""
        max_x = 15
        max_y = image_size[1] - datamatrix_size[1] - self.config.margin
        x = random.randint(self.config.margin, max(self.config.margin, max_x))
        y = random.randint(self.config.margin, max(self.config.margin, max_y))
        return (x, y)


class SyntheticDatasetGenerator:
    A4_WIDTH = 2480
    A4_HEIGHT = 3508
    VALID_IMAGE_FORMATS = ["PNG", "JPG", "JPEG"]

    def __init__(
        self,
        datamatrix_config: DataMatrixConfig = DataMatrixConfig(),
        base_dataset_name: Optional[str] = None,
        base_dataset: Optional[Dataset] = None,
        image_format: str = "PNG",
    ):

        if base_dataset is None and base_dataset_name is None:
            raise ValueError(
                "Either base_dataset or base_dataset_name must be provided"
            )
        if image_format not in self.VALID_IMAGE_FORMATS:
            raise ValueError(
                f"Invalid image format. Must be one of {self.VALID_IMAGE_FORMATS}"
            )

        self.base_dataset = (
            load_dataset(base_dataset_name) if base_dataset is None else base_dataset
        )
        self.datamatrix_gen = DataMatrixGenerator(datamatrix_config)

    def resize_to_a4(self, image):
        """Resize image to A4 size while maintaining aspect ratio and adding white padding"""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Calculate aspect ratios
        img_ratio = image.size[0] / image.size[1]
        a4_ratio = self.A4_WIDTH / self.A4_HEIGHT

        # Create new white background image
        new_img = Image.new("RGB", (self.A4_WIDTH, self.A4_HEIGHT), "white")

        # Calculate new dimensions maintaining aspect ratio
        if img_ratio > a4_ratio:
            # Image is wider than A4
            new_width = self.A4_WIDTH
            new_height = int(self.A4_WIDTH / img_ratio)
        else:
            # Image is taller than A4
            new_height = self.A4_HEIGHT
            new_width = int(self.A4_HEIGHT * img_ratio)

        # Resize original image
        resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Calculate position to paste (center)
        x = (self.A4_WIDTH - new_width) // 2
        y = (self.A4_HEIGHT - new_height) // 2

        # Paste resized image onto white background
        new_img.paste(resized_img, (x, y))

        return new_img

    def generate_sample(
        self, base_image: Image.Image
    ) -> Tuple[Image.Image, List[float]]:
        """Generate a single synthetic sample with bounding box."""
        new_image = base_image.copy()
        # Generate and resize DataMatrix
        barcode = self.datamatrix_gen.create_datamatrix()
        size = (
            self.datamatrix_gen.config.min_size,
            self.datamatrix_gen.config.max_size,
        )
        barcode = barcode.resize(size)

        # Get position and paste
        pos = self.datamatrix_gen.get_random_position(new_image.size, barcode.size)
        new_image.paste(barcode, pos)

        # Calculate normalized bounding box [x1, y1, x2, y2]
        x1 = pos[0] / new_image.width
        y1 = pos[1] / new_image.height
        x2 = (pos[0] + barcode.width) / new_image.width
        y2 = (pos[1] + barcode.height) / new_image.height

        return new_image, [x1, y1, x2, y2]

    def generate_dataset(
        self, num_samples: int, show_progress: bool = True
    ) -> List[Tuple[Image.Image, List[float]]]:
        """Generate multiple synthetic samples."""
        if num_samples > len(self.base_dataset):
            raise ValueError(
                f"Number of samples ({num_samples}) is greater than the base dataset size ({len(self.base_dataset)})."
            )

        samples = []
        iterator = tqdm(range(num_samples)) if show_progress else range(num_samples)
        for i in iterator:
            try:
                image_bytes = self.base_dataset[i]["images"][0]["bytes"]
                base_image = Image.open(io.BytesIO(image_bytes))
                base_image = self.resize_to_a4(base_image)
                sample = self.generate_sample(base_image)
                samples.append(sample)
            except Exception as e:
                print(f"Warning: Failed to generate sample {i}: {str(e)}")
                continue

        return samples


def create_hf_dataset(
    samples: List[Tuple[Image.Image, List[float]]], dataset_name: str
):
    """Create a Hugging Face dataset from the samples."""
    import datasets
    from PIL import Image
    import io

    def image_to_bytes(image):
        byte_arr = io.BytesIO()
        image.save(byte_arr, format="PNG")
        return byte_arr.getvalue()

    dataset = {
        "image": [image_to_bytes(img) for img, _ in samples],
        "bbox": [bbox for _, bbox in samples],
    }

    dataset = datasets.Dataset.from_dict(dataset)
    dataset = dataset.cast_column("image", datasets.Image())

    return dataset


if __name__ == "__main__":
    config = DataMatrixConfig(min_size=(40, 40), max_size=(80, 80))
    generator = SyntheticDatasetGenerator(config, base_dataset="mnist")  ## Placeholder
