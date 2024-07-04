import pandas as pd
import os

from . import utils
import pandas as pd
from dotenv import load_dotenv

import pandas as pd

import sys
import os
import argparse

# for tiltData v1.1.0
NL_COUNTRY_ID = "6ace185eedb813fe84c2eca7641f9fa0aa3bfdc3"


# ep_only = "source_1_ep"
# ci_only = "source_3_ci"
# ep_ci = "source_4_ep_ci"


def read_in_companyinfo_export(data_dir: str) -> pd.DataFrame:
    """Read in Company.info export files from the given data directory.

    Read in Company.info export files, select and rename relevant columns.

    Args:
        data_dir (str): Directory where we can find the export files.

    Returns:
        pd.DataFrame: Company.info in DataFrame with relevant columns.
    """

    # Columns to keep
    keep_cols = [
        "Kamer_van_Koophandel_nummer_12-cijferig",
        "Instellingsnaam",
        "Statutaire_naam",
        "Bedrijfsomschrijving",
        "Vestigingsadres_postcode",
        "Vestigingsadres_plaats",
        "SBI-code_locatie",
        "SBI-code_locatie_Omschrijving",
    ]

    print(f"Reading in company.info data files in {data_dir}")

    # list all entries in the directory
    list_dir = os.listdir(path=data_dir)

    read = []

    for entry in list_dir:
        filename = f"{data_dir}/{entry}"

        # sanity check: filter out folders
        if not os.path.isfile(filename):
            continue

        # sanity check: filter out incorrect file types
        if os.path.splitext(entry)[-1].lower() != ".xlsx":
            continue

        # file expected to be in the following format
        df = pd.read_excel(
            filename,
            dtype={
                "Kamer_van_Koophandel_nummer_12-cijferig": "str",
                "SBI-code_locatie": "str",
            },
        )[keep_cols]

        read.append(df)

    print(f"Read in {len(read)} data files")

    ci = pd.concat(read)

    # rename columns
    ci.rename(
        columns={
            "Kamer_van_Koophandel_nummer_12-cijferig": "company_id",
            "Instellingsnaam": "institution_name",
            "Statutaire_naam": "statutory_name",
            "Bedrijfsomschrijving": "company_description",
            "Vestigingsadres_postcode": "postcode",
            "Vestigingsadres_plaats": "place",
            "SBI-code_locatie_Omschrijving": "sbi_code_description",
            "SBI-code_locatie": "sbi_code",
        },
        inplace=True,
    )

    return ci


def merge_company_names(ci_df: pd.DataFrame) -> pd.DataFrame:
    """CompanyInfo has institution and statutory names. We keep statutory names
    where available, and institution name otherwise.

    Args:
        ci_df (pd.DataFrame): DataFrame of raw CompanyInfo data.

    Returns:
        pd.DataFrame: DataFrame of CompanyInfo data with the company name merged.
    """
    ci_df["company_name"] = ci_df.apply(
        lambda x: (
            x["statutory_name"].lower()
            if isinstance(x["statutory_name"], str)
            else x["institution_name"]
        ),
        axis=1,
    )

    ci_df.drop(columns=["institution_name", "statutory_name"], inplace=True)

    return ci_df


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


def create_dataset(data_dir="data/input/tiltData-v1.1.0"):

    # Company.info
    ci_df = pd.read_csv("data/processed/companyinfo.csv", dtype={"sbi_code": str})
    companies = ci_df[["company_id", "company_description"]]
    companies["country_un"] = "nl"

    # Europages
    tilt = read_csv(
        "companies", data_dir, ["companies_id", "information", "country_id"]
    )
    geography = read_csv(
        "geography", data_dir, ["country_id", "ecoinvent_geography", "priority"]
    )
    geography[geography["priority"] == 1]
    tilt = tilt.merge(geography, on="country_id")
    tilt = tilt.rename(columns={"ecoinvent_geography": "country_un"})

    # Matched
    match = pd.read_csv("data/processed/export.csv")
    match["source_id"] = "source_4_ep_ci"
    ep_ids = match.europages_company_id.tolist()
    ci_ids = match.companyinfo_company_id.tolist()

    drop_index = tilt[tilt["companies_id"].isin(ep_ids)].index
    tilt = tilt.drop(index=drop_index)

    tilt = tilt.rename(
        columns={"companies_id": "company_id", "information": "company_description"}
    )

    # Source ID
    tilt["source_id"] = "source_1_ep"
    companies["source_id"] = companies.apply(
        lambda x: "source_4_ep_ci" if x["company_id"] in ci_ids else "source_3_ci",
        axis=1,
    )

    companies = pd.concat([companies, tilt])

    products = read_csv("products", data_dir).rename(
        columns={"products_and_services": "product_name", "products_id": "product_id"}
    )
    ep_products_companies = read_csv("products_companies", data_dir).rename(
        columns={"companies_id": "company_id", "products_id": "product_id"}
    )

    ci_product = ep_products_companies.merge(
        match, left_on="company_id", right_on="europages_company_id", how="inner"
    )

    drop_index = ep_products_companies[
        ep_products_companies.company_id.isin(ep_ids)
    ].index
    ep_products_companies = ep_products_companies.drop(index=drop_index)

    products_companies = pd.concat(
        [ci_product[["company_id", "product_id"]], ep_products_companies]
    )

    # SBI activities
    sbi_activities = pd.read_csv("data/resources/sbi.csv", dtype={"SBI": str})
    sbi_activities = sbi_activities.rename(
        columns={"SBI": "sbi_code", "Omschrijving": "sbi_code_description"}
    )

    # Companies - SBI activities
    companies_sbi_activities = ci_df[["company_id", "sbi_code"]]

    save_dir = "data/dataset"
    companies.to_csv(f"{save_dir}/companies.csv", index=False)
    products.to_csv(f"{save_dir}/products.csv", index=False)
    products_companies.to_csv(f"{save_dir}/companies_products.csv", index=False)
    companies_sbi_activities.to_csv(
        f"{save_dir}/companies_sbi_activities.csv", index=False
    )
    sbi_activities.to_csv(f"{save_dir}/sbi_activities.csv", index=False)


def run_preprocessing(input_dir: str, save_dir: str):
    """Run preprocessing steps for Company.info data.

    Args:
        input_dir (str): Directory to find Company.info export files.
        save_dir (str): Directory to save preprocessed Company.info file.
    """
    # ci_df = read_in_companyinfo_export(input_dir)
    # ci_df = merge_company_names(ci_df)

    # save_filepath = f"{save_dir}/companyinfo.csv"
    # ci_df.to_csv(save_filepath, index=False)

    create_dataset()
