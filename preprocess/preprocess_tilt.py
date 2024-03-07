import pandas as pd
from dotenv import load_dotenv

import preprocess_utils as utils
import pandas as pd

import sys
import argparse

# for tiltData v1.1.0
NL_COUNTRY_ID = "6ace185eedb813fe84c2eca7641f9fa0aa3bfdc3"


def all_files_exist(data_dir):
    print("Checking that all the necessary files exist")
    if not utils.tilt_data_files_exist(data_dir):
        return False

    return True


def read_csv(filename, data_dir, columns=[]):
    path = f"{data_dir}/{filename}.csv"

    if len(columns) > 0:
        return pd.read_csv(path, usecols=columns)

    return pd.read_csv(path)


def format_postcode(postcode: str) -> str:
    """Format input Tilt postcode into '1234 AB' style for consistency.

    Args:
        postcode (str): Tilt postcode in

    Returns:
        str: Postcode in the format of '1234 AB'.
    """
    assert len(postcode) == 6, f"Input postcode is WRONG: {postcode}"

    num = postcode[:4]
    alph = postcode[4:].upper()
    return f"{num} {alph}"


def clean_postcode(tilt: pd.DataFrame) -> pd.DataFrame:
    """Clean up inconsistent postcode and city syntax

    Args:
        tilt (pd.DataFrame): DataFrame with 'postcode' and 'company_city' as columns.

    Returns:
        pd.DataFrame: Input DataFrame with 'postcode' and 'company_city' cleaned up.
    """

    # turn the columns into lists
    orig_postcode_list = tilt.postcode.to_list()
    orig_city_list = tilt.company_city.to_list()

    new_postcode_list = []
    new_city_list = []

    # iterate over the poscodes and cities
    for i, orig_postcode in enumerate(orig_postcode_list):
        orig_city = orig_city_list[i]

        # if city and postcode the same, just take one as reference
        if orig_postcode == orig_city:
            ref = orig_postcode

        # otherwise take concatenation as reference
        else:
            ref = f"{orig_postcode}{orig_city}"

        # if only city name, postcode is NA and and keep the city as is
        if ref in ["etten-leur", "kruiningen"]:
            new_postcode_list.append(pd.NA)
            new_city_list.append(ref)
            continue

        elif ref.isnumeric():
            continue

        # split "properly"
        orig_postcode, orig_city = ref.split(maxsplit=1)

        if "nederland" in orig_city:
            orig_city = orig_city.split(",")[0]

        new_postcode_list.append(format_postcode(orig_postcode))
        new_city_list.append(orig_city)

    tilt.postcode = new_postcode_list
    tilt.company_city = new_city_list

    return tilt


def get_ecoinvent_sectors(data_dir: str):
    ecoinvent_sectors = read_csv("sector_ecoinvent_delimited", data_dir)
    categories_ecoinvent_sectors = read_csv(
        "categories_sector_ecoinvent_delimited",
        data_dir,
        ["categories_id", "sector_ecoinvent_delimited_id"],
    )

    ecoinvent_sectors = pd.merge(
        categories_ecoinvent_sectors,
        ecoinvent_sectors,
        on="sector_ecoinvent_delimited_id",
    )

    return ecoinvent_sectors


def get_tilt_main_activity(data_dir: str):

    tilt_activity_list = [
        "agent/ representative",
        "distributor",
        "manufacturer/ producer",
        "retailer",
        "wholesaler",
    ]

    main_activity = read_csv(
        "main_activity", data_dir, columns=["main_activity_id", "main_activity"]
    )
    main_activity["main_activity"] = main_activity["main_activity"].apply(
        lambda x: x if x in tilt_activity_list else "other"
    )

    return main_activity


def merge_tilt_subsectors(dutch_companies):

    tilt_subsectors = dutch_companies[["companies_id", "tilt_subsector"]].to_dict(
        "records"
    )

    subsector_dict = {}

    for item in tilt_subsectors:
        company = item["companies_id"]
        subs = item["tilt_subsector"]

        if company in subsector_dict:
            if subs in subsector_dict[company]:
                continue
            else:
                subsector_dict[company].append(subs)
        else:
            subsector_dict[company] = [subs]

    subsectors = []

    for company in subsector_dict:
        subsectors.append(
            {"companies_id": company, "tilt_subsector": subsector_dict[company]}
        )

    df_tilt_subsectors = pd.DataFrame(subsectors)

    return df_tilt_subsectors


def get_tilt_sector_subsector(data_dir, res_dir):
    list_sector_subsector = read_csv(
        "EP_tilt_sector_mapper",
        res_dir,
        ["categories_id", "tilt_sector", "tilt_subsector"],
    )
    sector_to_companies = read_csv(
        "categories_companies", data_dir, ["categories_id", "companies_id"]
    )

    sector_to_companies = pd.merge(
        sector_to_companies, list_sector_subsector, on="categories_id"
    )

    return sector_to_companies


def read_tilt_data(data_dir, res_dir, save_dir):

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

    # load and merge tilt sector_subsector
    sector_to_companies = get_tilt_sector_subsector(data_dir, res_dir)
    dutch_companies = pd.merge(dutch_companies, sector_to_companies, on="companies_id")

    # load and merge ecoinvent sectors
    ecoinvent_sectors = get_ecoinvent_sectors(data_dir)
    dutch_companies = pd.merge(dutch_companies, ecoinvent_sectors, on="categories_id")

    # merge tilt subsectors into a list
    df_tilt_subsectors = merge_tilt_subsectors(dutch_companies)
    dutch_companies = utils.exclude_col(dutch_companies, ["tilt_subsector"])
    dutch_companies.drop_duplicates(inplace=True)

    dutch_companies = pd.merge(dutch_companies, df_tilt_subsectors, on="companies_id")

    # load and merge tilt list_products and services
    # list_products = read("products")
    # list_products_companies = read(
    #     "products_companies", ["products_id", "companies_id"]
    # )
    # dutch_companies = pd.merge(
    #     dutch_companies, list_products_companies, on="companies_id"
    # )

    # dutch_companies = pd.merge(dutch_companies, list_products, on="products_id")

    dutch_companies = utils.exclude_col(
        dutch_companies,
        [
            "categories_id",
            "products_id",
            "main_activity_id",
            "sector_ecoinvent_delimited_id",
        ],
    )

    dutch_companies = clean_postcode(dutch_companies)
    dutch_companies.to_csv(f"{save_dir}/tilt.csv", index=False)


def preprocess_tilt():
    return


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

    # check that the mapper folder exists

    if not all_files_exist(data_dir=args.data_dir):
        print("You're missing files to run preprocessing")
        sys.exit(0)

    read_tilt_data(args.data_dir, args.res_dir, args.save_dir)
