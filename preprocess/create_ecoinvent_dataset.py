from .match_companies import match_with_tilt, join_ecoinvent_tilt

from . import utils
import pandas as pd
import numpy as np

import json


def create_train_val_split(
    df: pd.DataFrame, val_split: float
) -> tuple[pd.DataFrame, pd.DataFrame]:

    list_index = df.index.tolist()
    np.random.shuffle(list_index)

    train_size = utils.calculate_train_size(dataset_size=len(df), val_split=val_split)

    train_index = list_index[:train_size]
    val_index = list_index[train_size:]

    train_df = df.loc[train_index]
    val_df = df.loc[val_index]

    return train_df, val_df


def get_multi_labels(df_input, df_labels, label_col):
    cid_tilt = df_input.companies_id_tilt

    def get_cid_rows(cid):
        return df_labels[df_labels.companies_id == cid]

    labels = [
        get_cid_rows(cid)[label_col].apply(lambda x: str(x)).unique().tolist()
        for cid in cid_tilt
    ]

    return labels


def create_orbis_ecoinvent_dataset(
    data_dir: str, save_dir: str, validation_split: float = 0.3
):
    """Create training, validation, test dataset

    Args:
        data_dir (str): _description_
        validation_split (float, optional): _description_. Defaults to 0.3.

    Returns:
        _type_: _description_
    """
    orbis_filename = f"{data_dir}/orbis.csv"
    orbis = pd.read_csv(orbis_filename)

    tilt_filename = f"{data_dir}/tilt.csv"
    tilt = pd.read_csv(tilt_filename)

    ecoinvent_filename = f"{data_dir}/ep_ei_companies.csv"
    ecoinvent = pd.read_csv(ecoinvent_filename)

    ecoinvent_tilt = join_ecoinvent_tilt(tilt, ecoinvent)

    orbis_tilt = match_with_tilt(ecoinvent_tilt, orbis)

    orbis_tilt_companies_id = orbis_tilt["companies_id_other"].unique()
    df_test_input = orbis[~orbis.companies_id.isin(orbis_tilt_companies_id)]

    df_train_input, df_val_input = create_train_val_split(orbis_tilt, validation_split)

    label_col = "activity_id_product_id"
    df_train_labels = get_multi_labels(df_train_input, ecoinvent_tilt, label_col)
    df_val_labels = get_multi_labels(df_val_input, ecoinvent_tilt, label_col)

    dataset_df = {
        "train": (df_train_input, df_train_labels),
        "val": (df_val_input, df_val_labels),
        "test": (df_test_input, None),
    }

    for split in dataset_df:
        df_input, df_labels = dataset_df[split]

        save_dataset(df_input, df_labels, split, save_dir)


def create_companyinfo_dataset(
    data_dir: str, save_dir: str, validation_split: float = 0.3
):
    """Create training, validation, test dataset

    Args:
        data_dir (str): _description_
        validation_split (float, optional): _description_. Defaults to 0.3.

    Returns:
        _type_: _description_
    """
    ci_filename = f"{data_dir}/companyinfo.csv"
    companyinfo = pd.read_csv(ci_filename)

    tilt_filename = f"{data_dir}/tilt_cpc_isic.csv"
    tilt = pd.read_csv(tilt_filename)

    companyinfo_tilt = match_with_tilt(tilt, companyinfo)

    df_train_input, df_val_input = create_train_val_split(
        companyinfo_tilt, validation_split
    )

    df_train_labels = get_multi_labels(df_train_input, tilt)
    df_val_labels = get_multi_labels(df_val_input, tilt)

    dataset_df = {
        "train": (df_train_input, df_train_labels),
        "val": (df_val_input, df_val_labels),
    }

    for split in dataset_df:
        df_input, df_labels = dataset_df[split]
        save_dataset(df_input, df_labels, split, save_dir)


def save_dataset(df_input, df_labels, split, save_dir):

    dataset = create_ecoinvent(df_input, df_labels)
    filename = f"{save_dir}/ci_ecoinvent_{split}_dataset.json"

    utils.write_json(filename, dataset)
    print(f"{split} dataset has {len(df_input)} rows")
    print(f"Saved {split} dataset in {filename}")


def create_ecoinvent(df, labels=None):
    x = utils.exclude_col(
        df,
        [
            "companies_id_tilt",
            "companies_id_other",
            "postcode",
        ],
    ).to_json(orient="records")

    x = json.loads(x)

    dataset = {"x": x}

    if labels:
        dataset["y"] = labels

        assert len(x) == len(labels), "Something went wrong with x and y"
    return dataset
