import os
from huggingface_hub import login
from datasets import load_from_disk, load_dataset, DatasetDict, Dataset
from src.data.datamatrix import (
    DataMatrixGenerator,
    DataMatrixConfig,
    SyntheticDatasetGenerator,
    create_hf_dataset,
)

login()

ds = load_from_disk("data/docmatix_subset")
sets = ["train", "val", "test"]

config = DataMatrixConfig(min_size=100, max_size=100)

synth_datasets = {}
for set in sets:
    generator = SyntheticDatasetGenerator(config, base_dataset=ds[set])
    samples = generator.generate_dataset(len(ds[set]))
    synth_datasets[set] = samples


train_dicts = [
    {"image": item[0], "labels": item[1]} for item in synth_datasets["train"]
]
val_dicts = [{"image": item[0], "labels": item[1]} for item in synth_datasets["val"]]
test_dicts = [{"image": item[0], "labels": item[1]} for item in synth_datasets["test"]]

synth_ds = DatasetDict(
    {
        "train": Dataset.from_list(train_dicts),
        "val": Dataset.from_list(val_dicts),
        "test": Dataset.from_list(test_dicts),
    }
)

synth_ds.save_to_disk("data/synth_datamatrix_docmatix_subset")
synth_ds.push_to_hub(
    "franperic/synth_datamatrix_docmatix_subset", private=True, token=None
)
