# import cv2
# from scipy.ndimage import label
# import numpy as np
# from PIL import Image
# import datasets
# from datasets import load_dataset
# import matplotlib.pyplot as plt

# ds = load_dataset("HuggingFaceM4/Docmatix", "images", split=datasets.ReadInstruction("train", from_=0, to=10_000, unit="abs"))

# check = next(iter(ds))
# check["images"][0].show()

# def filter_white_background(image):
#     img_array = np.array(image)

#     if len(img_array.shape) == 2:
#         # Convert to grayscale
#         intensity = img_array
#     else:
#         intensity = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]

#     # Calculate percentage of white pixels (threshold > 240)
#     white_threshold = 240
#     white_pixels = intensity > white_threshold
#     white_ratio = np.sum(white_pixels) / white_pixels.size

#     num_labels, labels = cv2.connectedComponents(white_pixels.astype(np.uint8))
#     if num_labels > 1:
#         label_sizes = np.bincount(labels.flatten())[1:]
#         largest_region_ratio = np.max(label_sizes) / white_pixels.size
#     else:
#         largest_region_ratio = 0

#     return white_ratio > 0.8 and largest_region_ratio > 0.6


# for i in range(50):
#     check = next(iter(ds))
#     print(filter_white_background(check["images"][0]))


# iter_ds = iter(ds)
# for i in range(20):
#     a = next(iter_ds)
#     print(i)
#     if not filter_white_background(a["images"][0]):
#         break
# a["images"][0].show()


# from pylibdmtx.pylibdmtx import encode
# import cv2
# import numpy as np

# def generate_datamatrix_image():
#     encoded = encode('Sample text')
#     # Convert to 3-channel array (RGB)
#     img_array = np.frombuffer(encoded.pixels, dtype=np.uint8)
#     img = img_array.reshape((encoded.height, encoded.width, 3))

#     # Convert to grayscale if needed
#     img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#     return img_gray


# # To test it and save:
# img = generate_datamatrix_image()
# cv2.imwrite('datamatrix.png', img)

from datasets import load_from_disk
from src.data.datamatrix import (
    DataMatrixGenerator,
    DataMatrixConfig,
    SyntheticDatasetGenerator,
    create_hf_dataset,
)

ds = load_from_disk("src/data/exp")

config = DataMatrixConfig(min_size=(40, 40), max_size=(80, 80))
generator = SyntheticDatasetGenerator(config, base_dataset=ds)

samples = generator.generate_dataset(10)

samples[7][0].show()

# import itertools
# from datasets import Dataset

# subset_ds = itertools.islice(ds, 100)
# ds_list = list(subset_ds)

# reg_dataset = Dataset.from_list(ds_list)
# reg_dataset.save_to_disk("src/data/exp")
# reg_dataset.to_parquet("src/data/exp.parquet")
