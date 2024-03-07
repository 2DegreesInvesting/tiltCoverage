from dotenv import load_dotenv

import argparse
import sys
import pandas as pd


def get_europages_ecoinvent_mapper(data_dir: str) -> pd.DataFrame:
    """Join EuroPages company (NL) table with ecoinvent mapper table

    Args:
        data_dir (str): Directory where the EuroPages and ecoinvent tables can be found

    Returns:
        pd.DataFrame: EuroPages joined with ecoinvent mapper table containing colums
            'companies_id' and 'activity_uuid_product_uuid'
    """

    ep_ei_mapper = pd.read_csv(f"{data_dir}/20231121_mapper_ep_ei.csv")
    ep_companies = pd.read_csv(f"{data_dir}/ep_companies_NL.csv")

    # generate primary key 'group_var' to join tables
    ep_companies["group_var"] = (
        ep_companies["clustered"]
        + "-.-"
        + ep_companies["country"]
        + "-.-"
        + ep_companies["main_activity"]
    )

    # join on primary key 'group_var'
    ep_ei_companies = pd.merge(ep_companies, ep_ei_mapper, on="group_var")

    # convert GPT certainty labels to numeric values
    completion_mapper = {"low": 1, "medium": 2, "high": 3}
    ep_ei_companies["completion"] = ep_ei_companies["completion"].apply(
        lambda x: completion_mapper[x]
    )

    # only keep rows with the highest certainty (highest may still be 'low')
    ep_ei_companies = pd.DataFrame(
        ep_ei_companies.groupby(["companies_id", "activity_uuid_product_uuid"])[
            "completion"
        ]
        .max()
        .reset_index()
    )

    # only keey id columns
    ep_ei_companies = ep_ei_companies[["companies_id", "activity_uuid_product_uuid"]]

    return ep_ei_companies


def get_ecoinvent_activity_product(ep_ei_companies: pd.DataFrame, data_dir: str):

    # get ecoinvent data table, rename columns
    ei_data = pd.read_csv(f"{data_dir}/ecoinvent_complete.csv").rename(
        columns={
            "Activity UUID & Product UUID": "activity_uuid_product_uuid",
            "Activity Name": "ei_activity",
            "Reference Product Name": "ei_product",
        }
    )

    # only keep relevant columns
    ei_data = ei_data[
        [
            "activity_uuid_product_uuid",
            "ei_activity",
            "ei_product",
        ]
    ]

    ep_ei_companies = pd.merge(
        ep_ei_companies, ei_data, on="activity_uuid_product_uuid"
    )

    return ep_ei_companies


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../input")
    parser.add_argument("--save_dir", type=str, default="../input/processed")

    # make sure that env file is there
    env_loaded = load_dotenv()
    if not env_loaded:
        print(
            "Your environment variables could not be loaded. Check that you have a .env file."
        )
        sys.exit(0)

    args = parser.parse_args()

    # get europages ecoinvent mapper
    print("Getting EuroPages-ecoinvent mapper")
    ep_ei_mapper = get_europages_ecoinvent_mapper(args.data_dir)

    # map ecoinvent activities and products to europages companies_id
    print("Mapping ecoinvent activities and products to EuroPages companies")
    ep_ei_companies = get_ecoinvent_activity_product(ep_ei_mapper, args.data_dir)
    ep_ei_companies.to_csv(f"{args.save_dir}/ep_ei_companies.csv", index=False)
