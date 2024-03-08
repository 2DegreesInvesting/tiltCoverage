from itertools import product
from difflib import SequenceMatcher
from dotenv import load_dotenv

import preprocess_utils as utils
import pandas as pd

import sys
import string
import argparse
import os
import json
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

    for index in names_dict:
        company = names_dict[index]
        name = company["company_name"].lower()
        postcode = company["postcode"]

        start_alphabet = name[0]

        if start_alphabet in alpha_dict:

            alpha_dict[start_alphabet].append((name, postcode, index))
        else:
            alpha_dict[start_alphabet] = [(name, postcode, index)]

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

    # find the common postcodes between the dataframes
    common_postcode = find_common_postcode(df_tilt, df_other)

    def get_company_info(df):
        df = df[df.postcode.isin(common_postcode)]
        names_dict = df[["company_name", "postcode"]].to_dict(orient="index")

        # DESIGN CHOICE: split the names into alphabetised list
        alphabetised = alphabetise_company_names(names_dict)
        return alphabetised

    names_dict_tilt = get_company_info(df_tilt)
    names_dict_other = get_company_info(df_other)

    alphabets = string.ascii_lowercase

    pairs = []
    for letter in alphabets:
        pairs.extend(create_all_pairs(names_dict_tilt, names_dict_other, letter))

    print("Finding matching companies")
    # find matching companies based on company name and postcode
    matched_pairs_idx = find_matching_pairs(pairs)

    companies_id_mapper = {}

    # retrieve companies_id of matched_pairs
    # TODO: can optimise by taking care of it in find_matching_pairs
    for tilt, other in matched_pairs_idx:
        idx_tilt = tilt[-1]
        idx_other = other[-1]
        id_tilt = df_tilt.at[idx_tilt, "companies_id"]
        id_other = df_other.at[idx_other, "companies_id"]

        companies_id_mapper[id_other] = id_tilt

    return companies_id_mapper


def pseudocode(df_tilt: pd.DataFrame, df_orbis: pd.DataFrame):

    tilt_postcode_groups = df_tilt.groupby("postcode").groups

    orbis_postcode = df_orbis.postcode.unique()

    # TODO what is the best way to keep track of the matches?
    matches = {}

    # iterate over tilt companies grouped by postcodes
    for postcode in tilt_postcode_groups:
        postcode_row_idx = tilt_postcode_groups[postcode]
        tilt_postcode_group = df_tilt.loc[postcode_row_idx]

        # tilt company does not exist in orbis, therefore no match found
        if postcode not in orbis_postcode:
            continue

        orbis_postcode_group = df_orbis[df_orbis.postcode == postcode]

        if len(orbis_postcode_group) == 1:
            # create pairs against the rows that have names with the same alphabet,
            # sim check that name passes, if so, it is a match!
            # sim check should include that the fist three letters also match!!!!!!!!!!!!!!!!!
            # TODO name check
            continue

        # there are multiple orbis companies with the same postcodes - let the many to many matching commence
        find_postcode_match(postcode, tilt_postcode_group, orbis_postcode_group)


def find_postcode_match(tilt_postcode, tilt_postcode_group, orbis_postcode_group):

    tilt_names = tilt_postcode_group.company_name.tolist()
    orbis_names = orbis_postcode_group.company_name.tolist()

    tilt_dict = alphabetise(tilt_names)
    tilt_alpha = list(tilt_dict.keys())

    orbis_dict = alphabetise(orbis_names, reference_list=tilt_alpha)

    for letter in tilt_alpha:
        name_pairs = create_pairs(tilt_dict[letter], orbis_dict[letter])
        measure_similarity(name_pairs)
        # check matches for the first three letters
        # if sim score greater than 0.9 then same company <- sanity check ish

    return


def alphabetise(list_names: list[str], reference_list=[]) -> dict:
    alphabet_dict = {}

    for name in list_names:
        letter = name.lower()[0]

        if len(reference_list) > 0 and letter not in reference_list:
            continue

        if letter in alphabet_dict:
            alphabet_dict[letter].append(name)
        else:
            alphabet_dict[letter] = [name]

    return alphabet_dict


def compute_sim_score(str_one, str_two):
    return


def measure_similarity(list_pairs, sim_threshold=0.9):
    match_found = 0

    for name_one, name_two in list_pairs:
        if name_one[:3] == name_two[:3]:
            # sanity check, if sim score
            if compute_sim_score >= sim_threshold:
                match_found += 1

    return


def create_pairs(list_one: list[str], list_two: list[str]) -> list[tuple[str]]:
    return
