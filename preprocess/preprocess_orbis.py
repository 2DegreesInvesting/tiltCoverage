from dotenv import load_dotenv

import preprocess_utils as utils
import pandas as pd

import os
import sys
import argparse


def all_files_exist(dir_dict: dict) -> bool:
    """Check if all necessary files exist.

    Args:
        dir_dict (dict): Keys are categories of files, values are folder name

    Returns:
        bool: True if all necessary files exist, False otherwise
    """
    print("Checking that all the necessary files exist")
    keys = ["data", "resource"]

    print("> Checking if all directory exists")
    if not all([utils.directory_exists(dir_dict[k]) for k in keys]):
        return False

    # create save directory if doesn't exist
    os.makedirs(dir_dict["save"], exist_ok=True)

    for k in keys:
        if k == "data":
            print("> Checking if input data exists")
            # checks if there is at least one data file in the data dir
            if not utils.orbis_data_files_exist(dir_dict["data"]):
                return False

        elif k == "resource":
            print("> Checking if additional resources exists")
            # checks if all mapper files exist in the mapper dir
            if not utils.orbis_resource_files_exist(dir_dict["resource"]):
                return False

    return True


def read_in_orbis_export(data_dir: str) -> list[pd.DataFrame]:
    """Read in Orbis export files from the specified data directory.

    Export files are expected to be in a specific format. The table should be in
    a sheet named 'Results', and data fields are expected to be contained in
    columns B to AN.

    Args:
        data_dir (str): Directory where we can find the export files.

    Returns:
        list[pd.DataFrame]: List of DataFrames read in from Orbis export files.
    """

    print(f"Reading in Orbis data files in {data_dir}")

    # list all entries in the directory
    list_dir = os.listdir(path=data_dir)

    read = []

    for entry in list_dir:
        filename = f"{data_dir}/{entry}"

        # sanity check: filter out folders
        if not os.path.isfile(filename):
            continue

        # sanity check: filter out incorrect file types
        if os.path.splitext(entry)[-1] != ".xlsx":
            continue

        # file expected to be in the following format
        df = pd.read_excel(filename, sheet_name="Results", usecols="B:AN")

        read.append(df)

    print(f"Read in {len(read)} data files")

    return read


def clean_orbis_export(list_df: list[pd.DataFrame], mapper_dir: str) -> pd.DataFrame:
    """Clean the raw orbis DataFrames by dropping empty/unnecessary rows and
    columns and combine into one DataFrame.

    Args:
        list_df (list[pd.DataFrame]): List of raw DataFrames.
        mapper_dir (str): Directory of mapper files.

    Returns:
        pd.DataFrame: One DataFrame
    """

    # company info is split into multiple "indented" rows
    # those extra rows do not contain information that we are interested in
    for raw_df in list_df:
        raw_df.dropna(
            axis=0,
            subset=[
                "City\nLatin Alphabet",
                "Company name Latin alphabet",
                "Primary code(s) in national industry classification",
                "Country",
            ],
            inplace=True,
        )

    # combine separate df into one
    orbis = pd.concat(list_df)

    name_mapper = utils.read_json(f"{mapper_dir}/orbis_col_renamer.json")
    cols_drop = utils.read_json(f"{mapper_dir}/orbis_col_dropper.json")

    # drop unnecessary columns
    orbis.drop(columns=cols_drop, inplace=True)

    # rename columns for readability
    orbis.rename(columns=name_mapper, inplace=True)

    return orbis


def create_companies_id(orbis_df: pd.DataFrame) -> pd.DataFrame:
    """Create unique id for each company in Orbis DataFrame in `companies_id` column

    Args:
        orbis_df (pd.DataFrame): Orbis DataFrame

    Returns:
        pd.DataFrame: Orbis DataFrame with new column `companies_id`
    """

    def create_uuid(company_name: str) -> str:
        # create md5 uuid based on the company name
        company_name = company_name.lower()

        return utils.make_md5_uuid(company_name)

    orbis_df["companies_id"] = orbis_df["company_name"].apply(lambda x: create_uuid(x))

    return orbis_df


def remove_source_name(orbis_df: pd.DataFrame) -> pd.DataFrame:
    """Remove source name from the `products_and_services` column of Orbis

    Args:
        orbis_df (pd.DataFrame): Orbis DataFrame

    Returns:
        pd.DataFrame: Orbis DataFrame with source removed from products_and_services
    """

    def remove_source(text):
        # sources are mentioned as [source: <name>]
        return text.split("[")[0].strip()

    orbis_df["products_and_services"] = orbis_df["products_and_services"].apply(
        remove_source
    )

    return orbis_df


def preprocess_orbis(data_dir: str, resource_dir: str, save_dir: str) -> pd.DataFrame:
    """Preprocess Orbis export data to be used for tilt.

    Args:
        data_dir (str): Directory where we can find the export files.
        resource_dir (str): Directory where we can find the mapper files.
        save_dir (str): Directory where we save the processed files.


    Returns:
        pd.DataFrame: Processed Orbis data.
    """
    # read in orbis data
    data = read_in_orbis_export(data_dir)

    # sanity check: if no data to be processed, then exit
    if len(data) == 0:
        print("No data to process, exiting programme. Bye!")
        return

    # clean up orbis export to exclude unnecessary rows, columns
    data = clean_orbis_export(data, resource_dir)

    # add companies_id
    data = create_companies_id(data)

    num_rows = len(data)
    print(f"Merged data table has {num_rows} rows")

    save_filename = f"{save_dir}/orbis.csv"
    print(f"Saving data to {save_filename}")
    data.to_csv(save_filename, index=False)

    return data


def run(data_dir: str, res_dir: str, save_dir: str):
    """Run preprocessing for Orbis

    Args:
        data_dir (str): Directory where the raw data files are
        res_dir (str): Directory where all resource files are
        save_dir (str): Directory where processed files should be saved
    """

    # check that all the necessary files exist
    dir_dict = {"data": data_dir, "resource": res_dir, "save": save_dir}
    if not all_files_exist(dir_dict=dir_dict):
        print("You're missing files to run preprocessing")
        sys.exit(0)

    preprocess_orbis(data_dir=data_dir, resource_dir=res_dir, save_dir=save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/input/orbis",
        help="Directory where the raw data files are",
    )
    parser.add_argument(
        "--res_dir",
        type=str,
        default="../data/resources",
        help="Directory where all resource files are",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../data/processed",
        help="Directory where processed files should be saved",
    )

    print("!!Preprocess Orbis!!")

    args = parser.parse_args()

    run(args.data_dir, args.res_dir, args.save_dir)
