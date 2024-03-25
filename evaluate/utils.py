import json


def read_labelled_data(data_dir: str, split: str):
    assert split in ["val", "train"], "Invalid data split"

    with open(f"{data_dir}/orbis_ecoinvent_{split}_dataset.json") as f:
        data = json.load(f)

    return data["x"], data["y"]
