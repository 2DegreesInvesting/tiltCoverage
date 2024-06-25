from haystack import Document

from typing import Optional, List, Any, Dict

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import time


from .retrievers.isic_retriever import TiltISICRetriever
from .retrievers.cpc_retriever import TiltCPCRetriever
from .retrievers.activity_retriever import TiltActivityRetriever


class TiltLedgerMapper:
    def __init__(self, provider, res_dir, doc_store_dir, top_k=5):

        # Initialise different retrievers
        self.isic_retriever = TiltISICRetriever(provider, res_dir, doc_store_dir, top_k)
        self.cpc_retriever = TiltCPCRetriever(provider, res_dir, doc_store_dir, top_k)
        self.activity_retriever = TiltActivityRetriever(provider, res_dir)

    def retrieve(self, companies, sbi_activities, companies_activities):
        print("> ISIC retrieval")
        # For each company, retrieve top 5 ISIC codes
        # TODO: threshold
        embeded_docs, isic_results = self.isic_retriever.retrieve(
            companies, sbi_activities, companies_activities
        )

        print("> CPC retrieval")
        # For each company, for the given ISIC codes, retrieve top 5 CPC codes
        # TODO: threshold
        cpc_results = self.cpc_retriever.retrieve(embeded_docs, isic_results)

        print("> Activity retrieval")
        activity_results = self.activity_retriever.retrieve(embeded_docs, isic_results)

        return isic_results, cpc_results, activity_results

    def map(self, isic_results, cpc_results, activity_results):
        # get tilt ledger entry id
        ledger = {}

        for company_id in cpc_results:

            company_isic = isic_results[company_id]
            combination = []
            for result in cpc_results[company_id]:
                cpc_code, isic_codes = result
                for isic in isic_codes:
                    if isic in company_isic:
                        for activity in activity_results[company_id][isic]:
                            combination.append((isic, cpc_code, activity, "nl"))

            ledger[company_id] = combination

        # mapper_table = join with ledger table

        return ledger


# results table:
# | company_id | isic | cpc | activity | geo |

# results.join(ledger, on=[isic, cpc, activity, geo])

# | company_id | ledger_entry_id|
