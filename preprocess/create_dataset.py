from match_companies import find_matching_companies
from dotenv import load_dotenv

from . import utils
import pandas as pd

import sys
import string
import argparse
import os
import json
import string


def create_dataset(data_dir: str, validation_split: float = 0.3):
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

    companies_id_mapper = find_matching_companies(tilt, orbis)

    def map_id(i):
        return companies_id_mapper[i]

    companies_id_orbis = list(companies_id_mapper.keys())
    tilt.rename(columns={"companies_id": "companies_id_tilt"}, inplace=True)

    orbis_val_df = orbis[orbis.companies_id.isin(companies_id_orbis)].drop_duplicates()
    orbis_val_df["companies_id_tilt"] = orbis_val_df["companies_id"].apply(map_id)

    orbis_val_df = pd.merge(tilt, orbis_val_df, on="companies_id_tilt")
    orbis_val_df.drop_duplicates(inplace=True)

    df_validation = orbis_val_df.sample(frac=validation_split)
    df_training = orbis_val_df[
        ~orbis_val_df.companies_id.isin(df_validation.companies_id.tolist())
    ]
    df_test = orbis[~orbis.companies_id.isin(companies_id_orbis)].drop_duplicates()

    print(f"Trianing set has {len(df_training)} rows")
    print(f"Validation set has {len(df_validation)} rows")
    print(f"Test set has {len(df_test)} rows")

    df_training.to_csv(f"{data_dir}/orbis_training.csv", index=False)
    df_validation.to_csv(f"{data_dir}/orbis_validation.csv", index=False)
    df_test.to_csv(f"{data_dir}/orbis_test.csv", index=False)

    return df_training, df_validation, df_test


def create_europages_validation_data(df_validation, res_dir: str):

    main_activity_mapper = utils.read_json(
        f"{res_dir}/tilt_main_activity_label_mapper.json"
    )
    sector_mapper = utils.read_json(f"{res_dir}/tilt_sector_label_mapper.json")
    subsector_mapper = utils.read_json(f"{res_dir}/tilt_subsector_label_mapper.json")

    val_x = utils.exclude_col(
        df_validation,
        [
            "companies_id",
            "main_activity",
            "tilt_sector",
            "tilt_subsector",
            "city",
            "postcode",
            "kvk_number",
        ],
    ).to_json(orient="records")
    val_x = json.loads(val_x)

    val_activity = df_validation["main_activity"].tolist()
    val_activity_y = (
        df_validation["main_activity"].apply(lambda x: main_activity_mapper[x]).tolist()
    )

    val_sector = df_validation["tilt_sector"].tolist()
    val_sector_y = (
        df_validation["tilt_sector"].apply(lambda x: sector_mapper[x]).tolist()
    )

    df_active_validation = df_validation[df_validation["main_activity"] != "other"]
    val_subsector = df_active_validation["tilt_subsector"].tolist()
    val_subsector_y = (
        df_active_validation["tilt_subsector"]
        .apply(lambda x: [subsector_mapper.get(subx, 100) for subx in eval(x)])
        .tolist()
    )

    val_dataset = {
        "x": val_x,
        "main_activity": {
            "y": val_activity_y,
            "y_text": val_activity,
        },
        "sector": {
            "y": val_sector_y,
            "y_text": val_sector,
        },
        "subsector": {
            "y": val_subsector_y,
            "y_text": val_subsector,
        },
    }

    return val_dataset


def create_ecoinvent_validation_data_separately(df_validation, data_dir):

    ep_ei_companies = pd.read_csv(f"{data_dir}/ep_ei_companies.csv")
    df_validation = pd.merge(df_validation, ep_ei_companies, on="companies_id")

    val_x = utils.exclude_col(
        df_validation,
        [
            "companies_id",
            "main_activity",
            "tilt_sector",
            "tilt_subsector",
            "city",
            "postcode",
            "kvk_number",
        ],
    ).to_json(orient="records")

    val_x = json.loads(val_x)
    val_activity = df_validation["ei_activity"].tolist()
    val_product = df_validation["ei_product"].tolist()

    val_dataset = {"x": val_x, "ei_activity": val_activity, "ei_product": val_product}

    return val_dataset


def create_ecoinvent(df, data_dir, labelled):

    if labelled:

        ep_ei_companies = pd.read_csv(f"{data_dir}/ep_ei_companies.csv").rename(
            columns={"companies_id": "companies_id_tilt"}
        )
        df = pd.merge(df, ep_ei_companies, on="companies_id_tilt").drop_duplicates()

    x = utils.exclude_col(
        df,
        [
            "companies_id",
            "main_activity",
            "tilt_sector",
            "tilt_subsector",
            "city",
            "postcode",
            "kvk_number",
        ],
    ).to_json(orient="records")

    x = json.loads(x)

    dataset = {"x": x}

    if labelled:
        y = df["activity_uuid_product_uuid"].tolist()
        dataset["y"] = y

        assert len(x) == len(y), "Something went wrong with x and y"
    return dataset


def create_europages_validation_data_orbis(df_validation, res_dir: str, save_dir: str):

    val_dataset = create_europages_validation_data(df_validation, res_dir=res_dir)
    utils.write_json(f"{save_dir}/orbis_europages_validation_dataset.json", val_dataset)


def create_ecoinvent_data_orbis(
    df_validation, df_training, df_test, data_dir: str, save_dir: str
):
    dataset_df = {"training": df_training, "validation": df_validation, "test": df_test}

    for split in dataset_df:

        df = dataset_df[split]

        if split == "test":
            labelled = False
        else:
            labelled = True

        dataset = create_ecoinvent(df, data_dir=data_dir, labelled=labelled)
        filename = f"{save_dir}/orbis_ecoinvent_{split}_dataset.json"

        utils.write_json(filename, dataset)

        print(f"Saved {split} dataset in {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["orbis"], default="orbis")
    parser.add_argument("--data_dir", type=str, default="../input/processed")
    parser.add_argument("--res_dir", type=str, default="../resources")
    parser.add_argument("--save_dir", type=str, default="../dataset")
    parser.add_argument("--split", action="store_true", default=False)

    # make sure that env file is there
    env_loaded = load_dotenv()
    if not env_loaded:
        print(
            "Your environment variables could not be loaded. Check that you have a .env file."
        )
        sys.exit(0)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.split:
        if args.source == "orbis":
            df_training, df_validation, df_test = create_dataset(args.data_dir)
    else:

        if args.source == "orbis":
            df_training = pd.read_csv(f"{args.data_dir}/orbis_training.csv")
            df_validation = pd.read_csv(f"{args.data_dir}/orbis_validation.csv")
            df_test = pd.read_csv(f"{args.data_dir}/orbis_test.csv")

    if args.source == "orbis":

        # create_europages_validation_data_orbis(
        #     df_validation, args.res_dir, args.save_dir
        # )

        create_ecoinvent_data_orbis(
            df_training, df_validation, df_test, args.data_dir, args.save_dir
        )
