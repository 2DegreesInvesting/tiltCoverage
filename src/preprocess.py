import pandas as pd
import os

import utils


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
            "Kamer_van_Koophandel_nummer_12-cijferig": "companies_id",
            "Instellingsnaam": "institution_name",
            "Statutaire_naam": "statutory_name",
            "Bedrijfsomschrijving": "description",
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


def run_preprocessing(input_dir: str, save_dir: str):
    """Run preprocessing steps for Company.info data.

    Args:
        input_dir (str): Directory to find Company.info export files.
        save_dir (str): Directory to save preprocessed Company.info file.
    """
    ci_df = read_in_companyinfo_export(input_dir)
    ci_df = merge_company_names(ci_df)
    ci_df.to_csv(f"{save_dir}/companyinfo.csv", index=False)

    ci_dataset = ci_df.to_dict("records")
    utils.write_json(f"{save_dir}/companyinfo_dataset.json", ci_dataset)
