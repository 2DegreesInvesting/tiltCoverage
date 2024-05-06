from .similarity import jaro_winkler
import pandas as pd


def find_matching_companies(
    df_tilt: pd.DataFrame, df_other: pd.DataFrame, similarity_threshold=0.98
) -> pd.DataFrame:

    tilt_unique = df_tilt.drop_duplicates(
        subset=["companies_id_tilt", "company_name", "postcode"]
    )
    other_unique = df_other.drop_duplicates(
        subset=["companies_id_other", "company_name", "postcode"]
    )

    join_postcode = tilt_unique.merge(
        other_unique, how="inner", on="postcode", suffixes=("_tilt", "_other")
    )

    # make all the company names lower case
    join_postcode["company_name_other"] = join_postcode["company_name_other"].apply(
        lambda x: x.lower() if isinstance(x, str) else x
    )

    # compute similarity scores
    join_postcode["similarity"] = join_postcode.apply(
        lambda x: jaro_winkler(x["company_name_tilt"], x["company_name_other"]), axis=1
    )

    # only keep the ones with score above threshold
    matched_companies = join_postcode[join_postcode.similarity >= similarity_threshold]

    return matched_companies[["companies_id_tilt", "companies_id_other"]]


def join_ecoinvent_tilt(df_tilt, df_ecoinvent):
    df_tilt = df_tilt[["companies_id", "company_name", "postcode"]]
    ecoinvent_tilt = pd.merge(df_ecoinvent, df_tilt, on="companies_id", how="left")

    return ecoinvent_tilt


def match_with_tilt(df_tilt: pd.DataFrame, df_other: pd.DataFrame) -> pd.DataFrame:

    df_tilt = df_tilt.rename(columns={"companies_id": "companies_id_tilt"})
    df_other = df_other.rename(columns={"companies_id": "companies_id_other"})

    matched_companies = find_matching_companies(df_tilt, df_other)

    matched_companies = matched_companies.join(
        df_tilt, on="companies_id_tilt", how="inner"
    )
    matched_companies = matched_companies.join(
        df_other, on="companies_id_other", how="inner"
    )

    return matched_companies
