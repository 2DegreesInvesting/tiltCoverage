from dotenv import load_dotenv

import pandas as pd
from . import preprocess_utils as utils

import argparse
import sys
import os


def all_files_exist(dir_dict) -> bool:
    """Check if all necessary files exist.

    Args:
        dir_dict (dict): Keys are categories of files, values are folder name

    Returns:
        bool: True if all necessary files exist, False otherwise
    """
    print("Checking that all the necessary files exist")

    # create save directory if doesn't exist
    os.makedirs(dir_dict["save"], exist_ok=True)

    # checks if all resource files exist in the resource dir
    return utils.ecoinvent_resource_files_exist(dir_dict["resource"])


def ignore_geography(ei_df: pd.DataFrame, res_dir: str):
    ei_df["ei_activity_id"] = ei_df["ei_activity"].apply(
        lambda x: utils.make_md5_uuid(x)
    )
    ei_df["ei_product_id"] = ei_df["ei_product"].apply(lambda x: utils.make_md5_uuid(x))

    ei_df["activity_id_product_id"] = [
        "-".join([row["ei_activity_id"], row["ei_product_id"]])
        for _, row in ei_df.iterrows()
    ]

    aipi_mapper = {}

    unique_aipi = ei_df.activity_id_product_id.unique()

    for aipi in unique_aipi:
        aupu_geo = ei_df[ei_df.activity_id_product_id == aipi][
            ["activity_uuid_product_uuid", "geo"]
        ].to_dict("records")

        aipi_mapper[aipi] = aupu_geo

    utils.write_json(f"{res_dir}/aipi_aupu_geo_mapper.json", aipi_mapper)

    return ei_df


def get_europages_ecoinvent_mapper(res_dir: str) -> pd.DataFrame:
    """Join EuroPages company (NL) table with ecoinvent mapper table

    Args:
        res_dir (str): Directory where the EuroPages and ecoinvent tables can be found

    Returns:
        pd.DataFrame: EuroPages joined with ecoinvent mapper table containing colums
            'companies_id' and 'activity_uuid_product_uuid'
    """

    ep_ei_mapper = pd.read_csv(f"{res_dir}/20231121_mapper_ep_ei.csv")
    # focus on NL only
    ep_ei_mapper = ep_ei_mapper[ep_ei_mapper.ep_country == "netherlands"]

    ep_companies = pd.read_csv(f"{res_dir}/ep_companies_NL.csv")

    # generate primary key 'group_var' to join tables
    ep_companies["group_var"] = (
        ep_companies["clustered"]
        + "-.-"
        + ep_companies["country"]
        + "-.-"
        + ep_companies["main_activity"]
    )

    # join on primary key 'group_var'
    ep_ei_mapper = pd.merge(ep_companies, ep_ei_mapper, on="group_var").dropna(
        subset=["completion"]
    )

    # convert GPT certainty labels to numeric values
    completion_mapper = {"low": 1, "medium": 2, "high": 3}
    ep_ei_mapper["completion"] = ep_ei_mapper["completion"].apply(
        lambda x: completion_mapper[x]
    )

    # only keep rows with the highest certainty (highest may still be 'low')
    # there are still at least one activity_uuid_product_uuid per company
    ep_ei_mapper = pd.DataFrame(
        ep_ei_mapper.groupby(["companies_id", "activity_uuid_product_uuid"])[
            "completion"
        ]
        .max()
        .reset_index()
    )

    # only keey id columns
    ep_ei_mapper = ep_ei_mapper[["companies_id", "activity_uuid_product_uuid"]]

    return ep_ei_mapper


def get_ecoinvent_data(res_dir: str) -> pd.DataFrame:
    ei_data = pd.read_csv(f"{res_dir}/ecoinvent_complete.csv").rename(
        columns={
            "Activity UUID & Product UUID": "activity_uuid_product_uuid",
            "Activity Name": "ei_activity",
            "Reference Product Name": "ei_product",
            "Activity UUID": "ei_activity_uuid",
            "Product UUID": "ei_product_uuid",
            "Product Information": "ei_product_info",
            "Geography": "geo",
        }
    )

    # only keep relevant columns
    ei_data = ei_data[
        [
            "activity_uuid_product_uuid",
            "ei_activity",
            "ei_product",
            "ei_activity_uuid",
            "ei_product_uuid",
            "ei_product_info",
            "geo",
        ]
    ]

    return ei_data


def get_ecoinvent_activity_product(
    ep_ei_mapper: pd.DataFrame, ecoinvent_df: pd.DataFrame, res_dir: str
) -> pd.DataFrame:
    """Get the ecoinvent activity_uuid_product_uuid for the EuroPages companies.

    Args:
        ep_ei_companies (pd.DataFrame): EuroPages ecoinvent table.
        data_dir (str): Directory where to find the ecoinvent activity and product file.

    Returns:
        pd.DataFrame: _description_
    """
    return pd.merge(ep_ei_mapper, ecoinvent_df, on="activity_uuid_product_uuid")


def preprocess_ecoinvent(res_dir: str, save_dir: str):

    # get europages ecoinvent mapper
    print("Getting EuroPages-ecoinvent mapper")
    ep_ei_mapper = get_europages_ecoinvent_mapper(res_dir)

    ei_df = get_ecoinvent_data(res_dir)

    # map ecoinvent activities and products to europages companies_id
    print("Mapping ecoinvent activities and products to EuroPages companies")
    # ep_ei_companies = get_ecoinvent_activity_product(ep_ei_mapper, ei_df, res_dir)
    ei_df = ignore_geography(ei_df, res_dir)
    ep_ei_companies = pd.merge(ep_ei_mapper, ei_df, on="activity_uuid_product_uuid")
    ep_ei_companies.drop(columns=["activity_uuid_product_uuid"], inplace=True)
    ei_df.drop(
        columns=["activity_uuid_product_uuid", "ei_activity_uuid", "ei_product_uuid"],
        inplace=True,
    )

    ei_df.drop_duplicates(subset=["activity_id_product_id"], inplace=True)

    ei_df.to_csv(f"{res_dir}/ecoinvent_complete_no_geo.csv", index=False)
    ep_ei_companies.to_csv(f"{save_dir}/ep_ei_companies.csv", index=False)


def run(res_dir: str, save_dir: str):
    print("Run preprocess ecoinvent")

    # check that all the necessary files exist
    dir_dict = {"resource": res_dir, "save": save_dir}
    if not all_files_exist(dir_dict=dir_dict):
        print("You're missing files to run preprocessing: ecoinvent")
        sys.exit(0)

    preprocess_ecoinvent(res_dir, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_dir", type=str, default="../input")
    parser.add_argument("--save_dir", type=str, default="../input/processed")

    # make sure that env file is there
    env_loaded = load_dotenv()
    if not env_loaded:
        print(
            "Your environment variables could not be loaded. Check that you have a .env file."
        )
        sys.exit(0)

    args = parser.parse_args()

    run(args.res_dir, args.save_dir)
