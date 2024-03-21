from itertools import product
from difflib import SequenceMatcher

from . import preprocess_utils as utils
import pandas as pd

import string


def find_common_postcode(df_one: pd.DataFrame, df_two: pd.DataFrame) -> list[str]:
    """Find a list of common postcodes between two DataFrames containing postcode.

    Args:
        df_one (pd.DataFrame): DataFrame that has a column 'postcode' in format '1234 AB'.
        df_two (pd.DataFrame): DataFrame that has a column 'postcode' in format '1234 AB'.

    Returns:
        list[str]: List of common postcodes between two DataFrames.
    """
    # get list of unique postcodes from each DataFrame
    postcode_one = df_one.postcode.unique()
    postcode_two = df_two.postcode.unique()

    # find the intersection -> common postcodes
    postcode_common = set(postcode_one) & set(postcode_two)

    print(f"Found {len(postcode_common)} postcodes in common")

    # return as list
    return list(postcode_common)


def alphabetise_company_names(names_dict: dict) -> dict:
    alpha_dict = {}

    for company in names_dict:
        companies_id = company["companies_id"]
        name = company["company_name"].lower()
        postcode = company["postcode"]

        start_alphabet = name[0]

        if start_alphabet in alpha_dict:

            alpha_dict[start_alphabet].append((name, postcode, companies_id))
        else:
            alpha_dict[start_alphabet] = [(name, postcode, companies_id)]

    return alpha_dict


def create_all_pairs(dict_one, dict_two, alphabet):
    no_alphabet_key = alphabet not in dict_one or alphabet not in dict_two

    if no_alphabet_key:
        return []

    no_name_values = len(dict_one[alphabet]) == 0 or len(dict_two[alphabet]) == 0
    if no_name_values:
        return []

    return list(product(dict_one[alphabet], dict_two[alphabet]))


def eliminate_pair(company_one: tuple, company_two: tuple) -> bool:
    """Preliminary check on whether to eliminate a pair based on some criteria:
    if the postcode or the first three letters of the names do not match,
    eliminate the pair as they are not a match.

    Args:
        company_one (tuple): Tuple of company name, postcode, and index in its original DataFrame.
        company_two (tuple): Tuple of compnay name, postcode, and index in its original DataFrame.

    Returns:
        bool: Whether to eliminate the pair.
    """
    name_one, postcode_one, _ = company_one
    name_two, postcode_two, _ = company_two

    # if different postcode, then not a match
    if postcode_one != postcode_two:
        return True

    # if first three letters of the names are not the same, not a match
    if name_one[:3] != name_two[:3]:
        return True

    return False


def find_matching_pairs(pair_list: list, threshold_score: float = 0.9) -> list:
    """Find matches of companies based on elimination criteria and similiarty
    in the company names.

    Args:
        pair_list (list): List of pairs of tuples, each tuple consists of company name, postcode, and index in its original DataFrame.
        threshold_score (float, optional): Lower bound threshold for the similarity score, similarity score must be higher than threshold to be considered a match. Defaults to 0.9.

    Returns:
        list: List of matched pairs.
    """
    elim = 0
    matches = []

    for t, o in pair_list:

        # if pair should already be eliminated, move on to next
        if eliminate_pair(t, o):
            elim += 1
            continue

        # create string matcher to check for name similarity
        matcher = SequenceMatcher(None, t[0], o[0])
        score = matcher.ratio()

        # if name similarity score above threshold score, consider a match
        if score > threshold_score:
            matches.append((t, o))

    if elim == len(pair_list):
        print(f">Eliminated all")
    else:
        kept = len(pair_list) - elim
        print(f">Found {kept} matches")

    return matches


def find_matching_companies(
    df_tilt: pd.DataFrame, df_other: pd.DataFrame
) -> pd.DataFrame:

    df_tilt = df_tilt.drop_duplicates(
        subset=["companies_id", "company_name", "postcode"]
    )
    df_other = df_other.drop_duplicates(
        subset=["companies_id", "company_name", "postcode"]
    )

    # find the common postcodes between the dataframes
    common_postcode = find_common_postcode(df_tilt, df_other)

    def alphabetise(df):
        df = df[df.postcode.isin(common_postcode)]
        df_records = df.to_dict(orient="records")

        # DESIGN CHOICE: split the names into alphabetised list
        alphabetised = alphabetise_company_names(df_records)
        return alphabetised

    names_dict_tilt = alphabetise(df_tilt)
    names_dict_other = alphabetise(df_other)

    alphabets = string.ascii_lowercase

    pairs = []
    for letter in alphabets:
        pairs.extend(create_all_pairs(names_dict_tilt, names_dict_other, letter))

    # find matching companies based on company name and postcode
    print("Finding matching companies")
    matched_companies_ids = find_matching_pairs(pairs)

    companies_id_mapper = {}

    # retrieve companies_id of matched_pairs
    # TODO: can optimise by taking care of it in find_matching_pairs
    for tilt, other in matched_companies_ids:
        companies_id_tilt = tilt[-1]
        companies_id_other = other[-1]

        companies_id_mapper[companies_id_other] = companies_id_tilt

    return companies_id_mapper


def join_ecoinvent_tilt(df_tilt, df_ecoinvent):
    df_tilt = df_tilt[["companies_id", "company_name", "postcode"]]
    ecoinvent_tilt = pd.merge(df_ecoinvent, df_tilt, on="companies_id", how="left")

    return ecoinvent_tilt


def join_orbis_tilt(df_ecoinvent: pd.DataFrame, df_orbis: pd.DataFrame) -> pd.DataFrame:

    companies_id_mapper = find_matching_companies(df_ecoinvent, df_orbis)

    df_ecoinvent = df_ecoinvent.rename(columns={"companies_id": "companies_id_tilt"})
    df_orbis = df_orbis.rename(columns={"companies_id": "companies_id_orbis"})

    df_orbis["companies_id_tilt"] = df_orbis["companies_id_orbis"].apply(
        lambda x: companies_id_mapper.get(x, pd.NA)
    )

    df_orbis.dropna(axis="index", subset="companies_id_tilt", inplace=True)

    return df_orbis
