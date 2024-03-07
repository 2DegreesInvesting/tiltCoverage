import json
import os
import uuid


def read_json(filename):
    with open(filename) as f:
        data = json.load(f)

    return data


def write_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def exclude_col(df, ignore):
    assert isinstance(ignore, list), "Incorrect list!"

    original = df.columns
    return df[[col for col in original if col not in ignore]]


def keep_col(df, keep):
    assert isinstance(keep, list), "Incorrect list!"

    original = df.columns
    return df[[col for col in original if col in keep]]


def directory_exists(directory):
    return os.path.exists(directory) and os.path.isdir(directory)


def orbis_data_files_exist(directory):
    data_dir = os.listdir(path=directory)
    counter = 0

    for entry in data_dir:
        filename = f"{directory}/{entry}"

        # sanity check: filter out folders
        if not os.path.isfile(filename):
            continue

        # sanity check: filter out incorrect file types
        if os.path.splitext(entry)[-1] != ".xlsx":
            continue

        counter += 1

    if counter == 0:
        return False

    return True


def orbis_mapper_files_exist(directory):
    data_dir = os.listdir(path=directory)
    filenames = ["orbis_col_renamer.json", "orbis_col_dropper.json"]
    counter = 0

    for entry in filenames:
        if entry in data_dir:
            counter += 1

    if counter != len(filenames):
        return False

    return True


def tilt_data_files_exist(directory):
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

    data_dir = os.listdir(path=directory)
    counter = 0

    for entry in filenames:
        if entry in data_dir:
            counter += 1

    if counter != len(filenames):
        return False

    return True


def make_md5_uuid(name: str) -> str:
    """Make a UUID using a SHA-1 hash of a namespace UUID and a name"""
    return uuid.uuid5(uuid.NAMESPACE_DNS, name)
