import os
import random
from huggingface_hub import login
from datasets import load_from_disk, load_dataset, DatasetDict, Dataset
from src.data.datamatrix import (
    DataMatrixGenerator,
    DataMatrixConfig,
    SyntheticDatasetGenerator,
    create_hf_dataset,
)

login()
ds = load_dataset("HuggingFaceM4/Docmatix", "images", streaming=True)

total_samples = 3600
train_size = 3000
val_size = 300
test_size = 300

shuffled_ds = ds["train"].shuffle(seed=42)
train_ds = shuffled_ds.take(train_size)
val_ds = shuffled_ds.skip(train_size).take(val_size)
test_ds = shuffled_ds.skip(train_size + val_size).take(test_size)

train_list = list(train_ds)
val_list = list(val_ds)
test_list = list(test_ds)


dataset = DatasetDict(
    {
        "train": Dataset.from_list(train_list),
        "val": Dataset.from_list(val_list),
        "test": Dataset.from_list(test_list),
    }
)

output_path = "data/docmatix_subset"
os.makedirs(output_path, exist_ok=True)
dataset.save_to_disk(output_path)

dataset.push_to_hub("franperic/docmatix_subset", private=True, token=None)
