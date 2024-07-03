from .retrieve import run

from typing import List

import pandas as pd


def read_data_tables(data_dir: str) -> List[pd.DataFrame]:
    """Read in data tables"""

    companies = pd.read_csv(f"{data_dir}/companies.csv", dtype={"company_id": "str"})

    sbi_activities = pd.read_csv(
        f"{data_dir}/sbi_activities.csv", dtype={"sbi_code": "str"}
    )

    products = pd.read_csv(f"{data_dir}/products.csv")

    companies_products = pd.read_csv(
        f"{data_dir}/companies_products.csv", dtype={"company_id": "str"}
    )

    companies_sbi_activities = pd.read_csv(
        f"{data_dir}/companies_sbi_activities.csv",
        dtype={"sbi_code": "str", "company_id": "str"},
    )

    return (
        companies,
        sbi_activities,
        companies_sbi_activities,
        products,
        companies_products,
    )


def run_ledger_mapping(
    provider: str, data_dir: str, res_dir: str, doc_store_dir: str, output_dir: str
):
    """Read input data tables, map to tilt ledger, and save."""

    # Read input data
    (
        companies,
        sbi_activities,
        companies_sbi_activities,
        products,
        companies_products,
    ) = read_data_tables(data_dir)

    # Intialise ledger mapper
    run(
        companies,
        sbi_activities,
        companies_sbi_activities,
        products,
        companies_products,
        provider,
        res_dir,
        doc_store_dir,
        top_k=3,
        save_dir=output_dir,
    )
