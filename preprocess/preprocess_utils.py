import json
import os
import uuid
import pandas as pd


def check_file_extension(filename: str, extension: str) -> None | RuntimeError:
    """Check if filename as the expected extension, otherwise raise RuntimeError"""

    ext = os.path.splitext(filename)[-1]
    if ext != extension:
        raise RuntimeError(f"Expected a {extension} file, got {ext}.")


def read_json(filename: str) -> dict | list:
    """Opens and loads JSON files"""

    check_file_extension(filename=filename, extension=".json")

    with open(filename) as f:
        data = json.load(f)

    return data


def write_json(filename: str, data: dict | list) -> None:
    """Creates a JSON file"""

    check_file_extension(filename=filename, extension=".json")

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def exclude_col(df: pd.DataFrame, exclude: list[str]) -> pd.DataFrame:
    """Return the DataFrame having excluded specified columns

    Args:
        df (pd.DataFrame): DataFrame from which to exclude the specified columns
        exclude (list[str]): Names of columns to exclude from the DataFrame

    Returns:
        pd.DataFrame: Input DataFrame with the columns excluded
    """

    assert isinstance(exclude, list), "Incorrect list!"

    original = df.columns
    return df[[col for col in original if col not in exclude]]


def keep_col(df: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
    """Specify which columns of the DataFrame to keep. Used instead of
    exclude_col if it's easier to specify what to keep than to exclude.

    Args:
        df (pd.DataFrame): DataFrame in which to keep the specified columns
        keep (list[str]): Names of columns to keep in the DataFrame

    Returns:
        pd.DataFrame: Input DataFrame with only the columns specified
    """
    assert isinstance(keep, list), "Incorrect list!"

    original = df.columns
    return df[[col for col in original if col in keep]]


def directory_exists(directory: str) -> bool:
    """Checks whether the given directory exists"""
    return os.path.exists(directory) and os.path.isdir(directory)


def orbis_data_files_exist(orbis_data_dir: str) -> bool:
    """Checks whether at least one Orbis export as Excel file exists in the given directory."""

    orbis_data_files = os.listdir(path=orbis_data_dir)
    counter = 0

    for file in orbis_data_files:
        filename = f"{orbis_data_dir}/{file}"

        # sanity check: filter out folders
        if not os.path.isfile(filename):
            continue

        # sanity check: filter out incorrect file types
        if os.path.splitext(file)[-1] != ".xlsx":
            continue

        counter += 1

    # if no file at all
    if counter == 0:
        return False

    return True


def orbis_resource_files_exist(orbis_res_dir: str) -> bool:
    """Checks whether the expected Orbis resource files exist in the given directory"""

    orbis_res_files = os.listdir(path=orbis_res_dir)

    # the files we expect in the directory
    filenames = ["orbis_col_renamer.json", "orbis_col_dropper.json"]
    counter = 0

    for entry in filenames:
        if entry in orbis_res_files:
            counter += 1

    # if not all the files exist
    if counter != len(filenames):
        return False

    return True


def tilt_data_files_exist(tilt_data_dir: str) -> bool:
    """Checks whether the expected tilt data files exist in the given directory"""
    filenames = [
        # "categories.csv",
        "categories_companies.csv",
        # "categories_sector_ecoinvent_delimited.csv",
        # "clustered.csv",
        # "clustered_delimited.csv",
        "companies.csv",
        # "country.csv",
        # "delimited.csv",
        # "delimited_products.csv",
        # "geography.csv",
        # "issues.csv",
        # "issues_companies.csv",
        "main_activity.csv",
        # "products.csv",
        # "products_companies.csv",
        # "sea_food.csv",
        # "sea_food_companies.csv",
        # "sector_ecoinvent.csv",
        # "sector_ecoinvent_delimited.csv",
        # "sector_ecoinvent_delimited_sector_ecoinvent.csv",
    ]

    tilt_data_files = os.listdir(path=tilt_data_dir)
    counter = 0

    for file in filenames:
        if file in tilt_data_files:
            counter += 1

    if counter != len(filenames):
        return False

    return True


def make_md5_uuid(name: str) -> str:
    """Make a UUID using a SHA-1 hash of a namespace UUID and a name"""
    return uuid.uuid5(uuid.NAMESPACE_DNS, name)
