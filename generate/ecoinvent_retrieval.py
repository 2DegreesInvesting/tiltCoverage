import pandas as pd
import os
import numpy as np
from haystack.nodes import EmbeddingRetriever
from haystack.document_stores import InMemoryDocumentStore, FAISSDocumentStore
import datetime


def get_company_info(info_dict):

    naics = info_dict["naics_core_descr"]
    nace = info_dict["nace_core_descr"]
    ps = info_dict["products_and_services"]

    info = [naics, nace]

    if ps and not isinstance(ps, float):
        info.append(ps)

    return "; ".join(info)


def prep_orbis_data(input_data: dict) -> dict:
    orbis_dict = {
        item["companies_id_tilt"]: get_company_info(item) for item in input_data
    }
    return orbis_dict


def get_orbis_data():

    orbis = pd.read_csv("./data/processed/orbis.csv").to_dict("records")[:30]

    orbis_dict = {item["company_name"]: get_company_info(item) for item in orbis}

    return orbis_dict


def get_ecoinvent_data():

    ecoinvent = pd.read_csv("./data/input/ecoinvent_complete.csv").rename(
        columns={
            "Reference Product Name": "ei_product",
            "Activity Name": "ei_activity",
            "Activity UUID & Product UUID": "activity_uuid_product_uuid",
        }
    )[["ei_product", "ei_activity", "activity_uuid_product_uuid"]]

    return ecoinvent


def get_retriever(document_store, ecoinvent_data, yes=True):

    retriever = EmbeddingRetriever(
        document_store=document_store,
        batch_size=32,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        model_format="sentence_transformers",
    )

    if yes:
        documents = [
            {
                "content": row["ei_activity"] + ";" + row["ei_product"],
                "meta": {"id": str(row["activity_uuid_product_uuid"])},
            }
            for _, row in ecoinvent_data.iterrows()
        ]

        document_store.write_documents(documents)

        document_store.update_embeddings(retriever)
    retriever.embed_documents(document_store)

    return retriever


def predict_ecoinvent(input_data, data_dir):
    print(">Load FAISSDocumentStore")
    document_store = FAISSDocumentStore.load(
        index_path=f"{data_dir}/document_store.faiss"
    )
    print(">Get retriever")
    retriever = get_retriever(document_store, None, yes=False)

    preds = []
    print(">Retrieve results")
    for company_id in input_data:
        query = input_data[company_id]

        result = retriever.retrieve(query)[0]

        preds.append(result.meta["id"])

    assert len(preds) == len(input_data), "Prediction went wrong"
    return preds


def get_results(orbis_data, ei_retriever):

    companies_id = []
    result_id = []
    retrieval_score = []

    for company_id in orbis_data:
        query = orbis_data[company_id]
        companies_id.append(company_id)

        result = ei_retriever.retrieve(query)[0]

        pred_id = result.meta["id"]
        result_id.append(pred_id)

        score = result.score
        retrieval_score.append(score)

    assert len(companies_id) == len(result_id), "Faulty retrieval list"

    result_dict = {
        "companies_id": companies_id,
        "activity_uuid_product_uuid": result_id,
        "retrieval_score": retrieval_score,
    }

    result_df = pd.DataFrame(data=result_dict)

    return result_df


def main():

    ecoinvent_data = get_ecoinvent_data()
    orbis_data = get_orbis_data()

    retrieval_results = {}

    document_store = FAISSDocumentStore(embedding_dim=384)
    retriever = get_retriever(document_store, ecoinvent_data)
    # retriever.save("./data/resources/ecoinvent_retriever.mdl")
    retrieval_results = get_results(orbis_data, retriever)
    document_store.save("./data/dataset/document_store.faiss")

    return retrieval_results


if __name__ == "__main__":
    main()
