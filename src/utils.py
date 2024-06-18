import json
import os
import uuid
import pandas as pd

from dotenv import load_dotenv

import sys


def check_file_extension(filename: str, extension: str):
    """Check if filename as the expected extension, otherwise raise RuntimeError"""

    ext = os.path.splitext(filename)[-1]
    if ext != extension:
        raise RuntimeError(f"Expected a {extension} file, got {ext}.")


def read_json(filename: str):
    """Opens and loads JSON files"""

    check_file_extension(filename=filename, extension=".json")

    with open(filename) as f:
        data = json.load(f)

    return data


def write_json(filename: str, data) -> None:
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


def files_exist_in_dir(directory: str, list_filenames: list[str]) -> bool:
    """Checks whether the list of files exist in the directory

    Args:
        directory (str): Directory to search
        list_filenames (list[str]): List of file names to look up

    Returns:
        bool: Whether the files exist in the directory
    """
    # get list of files in the directory
    list_dir = os.listdir(path=directory)

    # check that all the files in list_filenames are in the directory
    all_in_dir = all([True for file in list_filenames if file in list_dir])

    return all_in_dir


def make_md5_uuid(name: str) -> str:
    """Make a UUID using a SHA-1 hash of a namespace UUID and a name"""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))


def load_env_file():
    """Load .env file. If not loaded, exit programme."""
    env_loaded = load_dotenv()

    if not env_loaded:
        print(
            "Your environment variables could not be loaded. Check that you have a .env file."
        )
        sys.exit(0)
