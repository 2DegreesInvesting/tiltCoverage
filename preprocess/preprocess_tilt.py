import pandas as pd
from dotenv import load_dotenv

from . import utils
import pandas as pd

import sys
import os
import argparse

# for tiltData v1.1.0
NL_COUNTRY_ID = "6ace185eedb813fe84c2eca7641f9fa0aa3bfdc3"


def all_files_exist(data_dir, res_dir, save_dir) -> bool:
    """Checks that all the necessary tilt files exist"""
    # create save directory if doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    print("Checking that all the necessary files exist")
    if not utils.tilt_data_files_exist(data_dir):
        return False

    if not utils.ecoinvent_resource_files_exist(res_dir):
        return False

    return True


def read_csv(table_name: str, data_dir: str, columns: list[str] = []) -> pd.DataFrame:
    """Load CSV file as DataFrame.

    Args:
        table_name (str): Name of the table (name of the csv file without the extension)
        data_dir (str): Directory of the table data file.
        columns (list[str], optional): Columns to select from the table. Defaults to [].

    Returns:
        pd.DataFrame: Table loaded on to pandas DataFrame
    """

    # sanity check, in case extension is added to table name
    if os.path.splitext(table_name)[-1] == ".csv":
        raise RuntimeError(
            "Non-existent table name. Check that you don't include the extension."
        )

    filename = f"{data_dir}/{table_name}.csv"

    # if usecol columns are specified
    if len(columns) > 0:
        return pd.read_csv(filename, usecols=columns, dtype={"postcode": "str"})

    return pd.read_csv(filename, dtype={"postcode": "str"})


def format_postcode(postcode: str) -> str:
    """Format input Tilt postcode into '1234 AB' style for consistency with Orbis.

    Args:
        postcode (str): Tilt postcode in

    Returns:
        str: Postcode in the format of '1234 AB'.
    """

    # if postcode in 1234ab format
    if len(postcode) == 6:
        num = postcode[:4]
        alph = postcode[4:].upper()

    # else if postcode in 1234 ab format
    else:
        split = postcode.split()

        assert len(split) == 2, "Unrecgonised postcode format"

        num, alph = split

    postcode = f"{num} {alph}"

    return postcode


def check_postcode(orig_postcode, orig_city):
    """Clean up inconsistent postcode and city format

    Args:
        tilt (pd.DataFrame): DataFrame with 'postcode' and 'company_city' as columns.

    Returns:
        pd.DataFrame: Input DataFrame with 'postcode' and 'company_city' cleaned up.
    """
    # if city and postcode the same, just take one as reference
    if orig_postcode == orig_city:
        reference = orig_postcode

    # otherwise take concatenation as reference
    else:
        reference = f"{orig_postcode}{orig_city}"

    # if only city name, postcode is NA and and keep the city as is
    if reference in ["etten-leur", "kruiningen"]:
        return pd.NA

    elif reference.isnumeric():
        return pd.NA

    # split "properly"
    orig_postcode = reference.split(maxsplit=1)[0]

    return format_postcode(orig_postcode)


def get_tilt_main_activity(data_dir: str) -> pd.DataFrame:
    """Get the main activity table from tilt

    Args:
        data_dir (str): Directory in which the main activity table file is.

    Returns:
        pd.DataFrame: EuroPages main activity table as DataFrame.
    """

    # list of main activities that we are concerned with
    # NOTE: move to a file?
    ep_activity_list = [
        "agent/ representative",
        "distributor",
        "manufacturer/ producer",
        "retailer",
        "wholesaler",
    ]

    main_activity = read_csv(
        "main_activity", data_dir, columns=["main_activity_id", "main_activity"]
    )

    # set any other activity as other if not in the ep_activity_list
    main_activity["main_activity"] = main_activity["main_activity"].apply(
        lambda x: x if x in ep_activity_list else "other"
    )

    return main_activity


def preprocess_tilt(data_dir):

    # load tilt company data
    companies = read_csv("companies", data_dir)

    # filter dutch comapnies
    dutch_companies = companies[companies["country_id"] == NL_COUNTRY_ID]
    dutch_companies = dutch_companies[
        [
            "companies_id",
            "company_name",
            "main_activity_id",
            "information",
            "postcode",
            "company_city",
        ]
    ]

    # load and merge tilt _main_activity
    tilt_main_activity = get_tilt_main_activity(data_dir)
    dutch_companies = pd.merge(
        dutch_companies, tilt_main_activity, on="main_activity_id"
    )

    dutch_companies = utils.exclude_col(
        dutch_companies,
        [
            "categories_id",
            "products_id",
            "main_activity_id",
            "sector_ecoinvent_delimited_id",
        ],
    )

    dutch_companies.drop_duplicates(inplace=True)

    dutch_companies["postcode"] = dutch_companies.apply(
        lambda x: check_postcode(x["postcode"], x["company_city"]), axis=1
    )

    return dutch_companies


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
    # completion_mapper = {"low": 1, "medium": 2, "high": 3}
    # ep_ei_mapper["completion"] = ep_ei_mapper["completion"].apply(
    # lambda x: completion_mapper[x]
    # )

    # only keep rows with the highest certainty (highest may still be 'low')
    # there are still at least one activity_uuid_product_uuid per company
    # ep_ei_mapper = pd.DataFrame(
    #     ep_ei_mapper.groupby(["companies_id", "activity_uuid_product_uuid"])[
    #         "completion"
    #     ]
    #     .max()
    #     .reset_index()
    # )

    ep_ei_mapper = ep_ei_mapper[ep_ei_mapper["completion"] == "high"]

    # only keey id columns
    ep_ei_mapper = ep_ei_mapper[["companies_id", "activity_uuid_product_uuid"]]

    return ep_ei_mapper


def get_ecoinvent_data(res_dir: str) -> pd.DataFrame:
    ei_data = pd.read_csv(f"{res_dir}/ecoinvent_complete.csv").rename(
        columns={
            "Activity UUID & Product UUID": "activity_uuid_product_uuid",
            "ISIC Classification": "isic_descr",
            "isic_4digit": "isic_code",
            "ISIC Section": "isic_section",
            "CPC Classification": "cpc",
            "Geography": "geo",
        }
    )

    # only keep relevant columns
    ei_data = ei_data[
        [
            "activity_uuid_product_uuid",
            "isic_code",
            "isic_descr",
            "cpc",
            "isic_section",
        ]
    ]

    return ei_data


def clean_cpc_isic_col(df):
    df["cpc_code"] = df["cpc"].apply(lambda x: x.split(":")[0].strip())
    df["cpc_descr"] = df["cpc"].apply(lambda x: x.split(":", maxsplit=1)[-1].strip())

    df["isic_descr"] = df["isic_descr"].apply(
        lambda x: x.split(":", maxsplit=1)[-1].strip()
    )

    return df.drop(columns=["isic_code", "isic_section", "cpc"])


def preprocess_ecoinvent(res_dir: str):

    # get europages ecoinvent mapper
    print("Getting EuroPages-ecoinvent mapper")
    ep_ei_mapper = get_europages_ecoinvent_mapper(res_dir)

    ei_df = get_ecoinvent_data(res_dir)

    # map ecoinvent activities and products to europages companies_id
    print("Mapping ecoinvent activities and products to EuroPages companies")

    ep_ei_companies = pd.merge(ei_df, ep_ei_mapper, on="activity_uuid_product_uuid")
    ep_ei_companies.drop(columns=["activity_uuid_product_uuid"], inplace=True)

    ep = clean_cpc_isic_col(ep_ei_companies)

    return ep


def run(data_dir: str, res_dir: str, save_dir: str):
    """Run preprocessing for tilt

    Args:
        data_dir (str): Directory where the raw data files are
        res_dir (str): Directory where all resource files are
        save_dir (str): Directory where processed files should be saved
    """

    # check that the necessary files exist
    if not all_files_exist(data_dir, res_dir, save_dir):
        print("You're missing files to run preprocessing")
        sys.exit(0)

    ep = preprocess_tilt(data_dir)
    ei = preprocess_ecoinvent(res_dir=res_dir)

    ep = pd.merge(ep, ei, on="companies_id", how="inner")
    ep = ep.drop_duplicates()

    ep.to_csv(f"{save_dir}/tilt_cpc_isic.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/input/tiltData-v1.1.0")
    parser.add_argument("--res_dir", type=str, default="../data/resources")
    parser.add_argument("--save_dir", type=str, default="../data/processed")

    # make sure that env file is there
    env_loaded = load_dotenv()
    if not env_loaded:
        print(
            "Your environment variables could not be loaded. Check that you have a .env file."
        )
        sys.exit(0)

    args = parser.parse_args()

    run(args.data_dir, args.res_dir, args.save_dir)
