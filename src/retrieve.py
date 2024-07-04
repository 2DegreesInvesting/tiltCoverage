from haystack import Document

from typing import Optional, List, Any, Dict

import pandas as pd
import numpy as np
import datetime

from haystack.components.embedders import HuggingFaceAPIDocumentEmbedder
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.utils import Secret

from .retrievers.isic_retriever import run as isic_run
from .retrievers.cpc_retriever import run as cpc_run
from .retrievers.activity_retriever import TiltActivityRetriever
import pickle
import os

from tqdm import tqdm
from .utils import write_json, read_json


# results table:
# | company_id | isic | cpc | activity | geo |

# results.join(ledger, on=[isic, cpc, activity, geo])

# | company_id | ledger_entry_id|


def initialise(provider: str, res_dir: str):

    # self.activity_retriever = TiltActivityRetriever(provider, res_dir)

    isic_mapper = pd.read_csv(
        f"{res_dir}/ISIC_rev_4.csv",
        dtype={"Code": str, "Description": str},
    ).to_dict("records")
    isic_mapper = {item["Code"]: item["Description"] for item in isic_mapper}

    cpc_mapper = pd.read_csv(
        f"{res_dir}/cpc_ver_2_1.csv", dtype={"Code": str, "Description": str}
    ).to_dict("records")
    cpc_mapper = {item["Code"]: item["Description"] for item in cpc_mapper}

    # Initialise embedders to embed incoming queries for retrieval
    if provider == "openai":
        model_name = "text-embedding-3-small"
        env_key = "OPENAI_API_KEY"

        embedder = OpenAIDocumentEmbedder(
            api_key=Secret.from_env_var(env_key),
            model=model_name,
        )
    else:
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        env_key = "HUGGINGFACEHUB_API_TOKEN"

        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type="serverless_inference_api",
            api_params={"model": model_name},
            token=Secret.from_env_var(env_key),
        )

    return embedder, isic_mapper, cpc_mapper


def process_data_tables(
    companies,
    sbi_activities,
    companies_sbi_activities,
    products,
    companies_products,
):
    def concatenate_columns(row):
        values = [
            row["sbi_code_description"],
            row["company_description"],
            row["product_name"],
        ]
        concatenated = ";".join(filter(pd.notna, values))
        return concatenated if concatenated else "Not enough information available"

    companies = companies[companies.source_id.isin(["source_3_ci", "source_4_ep_ci"])]
    companies = companies[
        ["company_id", "company_description", "country_un"]
    ].drop_duplicates(subset=["company_id"])
    companies = companies.merge(companies_sbi_activities, on="company_id", how="left")
    companies = companies.merge(sbi_activities, on="sbi_code", how="left")

    companies_products = companies_products.merge(products, on="product_id", how="left")
    companies_products = (
        companies_products.groupby("company_id")["product_name"]
        .agg(product_name=", ".join)
        .reset_index()
    )

    companies = companies.merge(companies_products, on="company_id", how="left")

    companies["query"] = companies.apply(
        concatenate_columns,
        axis=1,
    )

    companies["isic_section"] = companies.sbi_code.apply(
        lambda x: "None" if pd.isna(x) else x[:2]
    )

    return companies


def get_query_documents(companies: pd.DataFrame) -> List[Document]:
    query_documents = [
        Document(
            content=company["query"],
            meta={
                "company_id": company["company_id"],
                "isic_section": company["isic_section"],
            },
        )
        for _, company in companies.iterrows()
    ]

    return query_documents


def embed_documents(embedder, query_documents: List[Document]) -> List[Document]:
    if os.path.isfile("embedded_doc.pkl"):
        print("> Loading embedding")
        with open("embedded_doc.pkl", "rb") as f:
            embedded = pickle.load(f)

    else:
        print("> Creating and saving embedding")
        embedded = [doc for doc in embedder.run(query_documents)["documents"]]
        with open("embedded_doc.pkl", "wb") as f:
            pickle.dump(embedded, f)

    return embedded


def get_country(companies: pd.DataFrame) -> Dict[str, str]:
    """For each company, get the country of operation"""

    companies = companies[["company_id", "country_un"]].to_dict("records")
    companies = {row["company_id"]: row["country_un"] for row in companies}

    return companies


def run(
    companies: pd.DataFrame,
    sbi_activities: pd.DataFrame,
    companies_sbi_activities: pd.DataFrame,
    products: pd.DataFrame,
    companies_products: pd.DataFrame,
    provider,
    res_dir,
    doc_store_dir,
    top_k,
    save_dir,
):
    # Preprocess the data source
    companies = process_data_tables(
        companies,
        sbi_activities,
        companies_sbi_activities,
        products,
        companies_products,
    )

    del sbi_activities, companies_sbi_activities, products, companies_products
    embedder, isic_mapper, cpc_mapper = initialise(provider, res_dir)

    predictions = predict(provider, res_dir, companies, embedder, doc_store_dir, top_k)

    mapping = map_to_ledger(predictions, isic_mapper, cpc_mapper)

    to_csv(mapping, save_dir)


def predict(provider, res_dir, companies, embedder, doc_store_dir, top_k):

    query_documents = get_query_documents(companies)
    print(len(query_documents))
    # query_documents = None
    embedded_documents = embed_documents(embedder, query_documents)

    print("Hello?")
    query_doc_ids = [doc.meta["company_id"] for doc in query_documents]
    print("yes")
    embedded_documents = [
        doc
        for doc in tqdm(embedded_documents)
        if doc.meta["company_id"] in query_doc_ids
    ]

    print("done with that")

    print("> ISIC retrieval")

    isic_results = isic_run(embedded_documents, provider, res_dir, doc_store_dir, top_k)

    # directory = os.fsencode(".")

    # isic_results = {}
    # for file in os.listdir(directory):
    #     filename = os.fsdecode(file)
    #     if filename.startswith("isic_partial_results") and filename.endswith(".json"):
    #         isic_results.update(read_json(filename))

    # isic_results = read_json("isic_results.json")

    print("> CPC retrieval")
    # For each company, for the given ISIC codes, retrieve top 5 CPC codes
    # TODO: threshold
    cpc_results = cpc_run(
        embedded_documents, isic_results, provider, res_dir, doc_store_dir, top_k
    )

    # write_json("cpc_results.json", cpc_results)
    # cpc_results = read_json("cpc_results.json")

    print("> Activity retrieval")
    activity_retriever = TiltActivityRetriever(provider, res_dir)
    activity_results = activity_retriever.retrieve(embedded_documents, isic_results)

    country_results = get_country(companies)

    predictions = {
        "queries": query_documents,
        "isic": isic_results,
        "cpc": cpc_results,
        "activity": activity_results,
        "geo": country_results,
    }
    return predictions


def map_to_ledger(predictions, isic_mapper, cpc_mapper):
    queries = predictions["queries"]

    queries = {doc.meta["company_id"]: doc for doc in queries}
    isic_results = predictions["isic"]
    cpc_results = predictions["cpc"]
    activity_results = predictions["activity"]
    country_results = predictions["geo"]

    mapping_data = []

    print("Mapping to ledger")
    # for each company id
    for company_id in cpc_results:

        query_doc = queries.get(company_id, None)

        if query_doc is None:
            continue
        else:
            query = query_doc.content
            query_isic = query_doc.meta["isic_section"]

        # get all the isics for company
        isic_preds = isic_results[company_id]["preds"]
        isic_scores = isic_results[company_id]["scores"]

        # for each cpc result
        for i in range(len(cpc_results[company_id]["preds"])):
            # these are the acceptable combinations of cpc and isic
            cpc_pred, corr_isic = cpc_results[company_id]["preds"][i]
            cpc_score = cpc_results[company_id]["scores"][i]

            for j in range(len(isic_preds)):

                for activity in activity_results[company_id]:
                    mapping_data.append(
                        (
                            company_id,
                            query,
                            query_isic,
                            isic_preds[j],
                            isic_mapper.get(isic_preds[j], "could not find"),
                            isic_scores[j],
                            cpc_pred,
                            cpc_mapper.get(cpc_pred, "could not find"),
                            cpc_score,
                            activity,
                            country_results[company_id],
                        )
                    )

    # mapper_table = join with ledger table

    mapping = pd.DataFrame(
        data=mapping_data,
        columns=[
            "company_id",
            "query",
            "query_isic_section",
            "isic_code",
            "isic_description",
            "isic_score",
            "cpc_code",
            "cpc_description",
            "cpc_score",
            "activity",
            "geo",
        ],
    )

    return mapping


def simplify_output(mapping):
    return mapping[
        [
            "company_id",
            "isic_code",
            "isic_score",
            "cpc_code",
            "cpc_score",
            "activity",
            "geo",
        ]
    ]


def to_csv(mapping, save_dir):

    current_date = datetime.datetime.now().strftime("%Y%m%d")
    ledger_mapping_filepath = f"{save_dir}/{current_date}_ledger_results.csv"
    ledger_mapping_verbose_filepath = f"{save_dir}/{current_date}_ledger_details.csv"

    mapping.to_csv(ledger_mapping_verbose_filepath, index=False)

    simple_mapping = simplify_output(mapping)
    simple_mapping.to_csv(ledger_mapping_filepath, index=False)
