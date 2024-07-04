import json
from typing import Dict, List
from . import utils
import pandas as pd

import numpy as np


def structure_isic_output(res_dir, preds, scores):
    isic = pd.read_csv(
        f"{res_dir}/ISIC_Rev_4_english_structure.Txt", dtype={"Code": "str"}
    )
    isic.drop_duplicates(inplace=True)
    isic = isic.set_index("Code")
    isic = isic.to_dict()["Description"]

    preds_descr = [": ".join([p, isic.get(p, "???")]) for p in preds]

    output = [[preds_descr[i], scores[i]] for i in range(len(preds))]
    return output


def structure_cpc_output(res_dir, preds, scores):
    cpc = pd.read_csv(
        f"{res_dir}/cpc_list.csv",
        dtype={"cpc": "str"},
        usecols=["cpc", "description"],
    )

    cpc.drop_duplicates(inplace=True)

    cpc = cpc.set_index("cpc")

    cpc = cpc.to_dict()["description"]

    preds_descr = [": ".join([p, cpc[p]]) for p in preds]

    output = [[preds_descr[i], scores[i]] for i in range(len(preds))]
    return output


def run_manual_inspection(data_dir, res_dir, output_dir):
    print(">Reading results")
    data = pd.read_csv("data/output/20240704_ledger_details.csv")

    # select random examples
    index = np.random.choice(data.index, 10, replace=False)
    selection = data.loc[index]

    letizia = data[["company_id", "isic_code", "cpc_code", "activity", "geo"]]
    letizia.rename(
        {
            "isic_code": "ISIC_4digit",
            "cpc_code": "CPC_Code",
            "activity": "Activity_Type",
            "geo": "Geography",
        }
    )

    letizia.to_csv("data/output/letizia.csv", index=False)

    selection.to_csv("data/output/inspection.csv", index=False)
