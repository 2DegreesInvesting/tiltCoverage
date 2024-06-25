import json
from typing import Dict, List
import pandas as pd

# from .retrieve import LedgerMapper
from .retrieve_new import TiltLedgerMapper

from . import utils
import numpy as np


def read_input_data(data_dir: str):

    companies = pd.read_csv(f"{data_dir}/companies.csv")
    sbi_activities = pd.read_csv(f"{data_dir}/sbi_activities.csv")
    companies_sbi_activities = pd.read_csv(f"{data_dir}/companies_sbi_activities.csv")

    return companies, sbi_activities, companies_sbi_activities


def run_ledger_mapping(provider, data_dir, res_dir, output_dir):
    print(">Reading input data")
    companies, sbi_activities, companies_sbi_activities = read_input_data(data_dir)[:10]

    print(">Initialise retriever")
    retriever = TiltLedgerMapper("openai", res_dir, "data/doc_store")

    isic, cpc, activity = retriever.retrieve(
        companies, sbi_activities, companies_sbi_activities
    )

    ledger = retriever.map(isic, cpc, activity)
    print(ledger)
