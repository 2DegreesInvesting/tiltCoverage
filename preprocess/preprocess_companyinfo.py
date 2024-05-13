from typing import List
import os
import pandas as pd


def read_in_companyinfo_export(data_dir: str) -> pd.DataFrame:
    """Read in Company.info export files from the given data directory.

    Args:
        data_dir (str): Directory where we can find the export files.

    Returns:
        pd.DataFrame:
    """

    # Columns to keep
    keep_cols = [
        "Kamer_van_Koophandel_nummer_12-cijferig",
        "Instellingsnaam",
        "Statutaire_naam",
        "Bedrijfsomschrijving",
        "Vestigingsadres_postcode",
        "Vestigingsadres_plaats",
        "SBI-code_Omschrijving",
        "SBI-code_2-cijferig_Omschrijving",
        "SBI-code_locatie_Omschrijving",
        "SBI-code_2-cijferig_locatie_Omschrijving",
        "SBI-code_segment_locatie_Omschrijving",
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
            filename, dtype={"Kamer_van_Koophandel_nummer_12-cijferig": "str"}
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
            "SBI-code_Omschrijving": "sbi",
            "SBI-code_2-cijferig_Omschrijving": "sbi_division",
            "SBI-code_locatie_Omschrijving": "sbi_branch",
            "SBI-code_2-cijferig_locatie_Omschrijving": "sbi_branch_division",
        },
        inplace=True,
    )

    return ci


def clean_companyinfo_export(ci_df) -> pd.DataFrame:
    ci_df["company_name"] = ci_df.apply(
        lambda x: (
            x["statutory_name"]
            if isinstance(x["statutory_name"], str)
            else x["institution_name"]
        ),
        axis=1,
    )

    ci_df.drop(columns=["institution_name", "statutory_name"], inplace=True)

    ci_df["isic"] = ci_df["sbi"].apply(lambda x: x[:4])

    return ci_df


def run(data_dir, save_dir):
    ci_df = read_in_companyinfo_export(data_dir)
    ci_df = clean_companyinfo_export(ci_df)
    ci_df.to_csv(f"{save_dir}/companyinfo.csv", index=False)
