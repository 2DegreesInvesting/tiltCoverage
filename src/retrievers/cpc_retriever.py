from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack import Document

from ..utils import MyChromaDocumentStore, write_json

from typing import List, Any, Dict

from tqdm import tqdm
import pandas as pd
import os


def get_document_store(
    provider: str, res_dir: str, doc_store_dir: str, env_key: str, model_name: str
) -> ChromaDocumentStore:
    """Create a new or load existing document store with CPC code and descriptions

    Args:
        res_dir (str): Directory to find the CPC table
        doc_store_dir (str): Directory where to save/load doc store

    Returns:
        ChromaDocumentStore: DocumentStore of CPC code and descriptions
    """

    doc_store_dir = f"{doc_store_dir}/cpc/{provider}"
    doc_store_path = f"{doc_store_dir}/chroma.sqlite3"

    # If doc_store file exists, then no need to write the documents
    if os.path.isfile(doc_store_path):
        print(">Load existing CPC doc store")
        documents = None

    # Else, create documents out of each CPC code and description
    else:
        print("> Create CPC doc store")
        cpc_list = pd.read_csv(
            f"{res_dir}/CPC_ver_2_1.csv",
            dtype={"Code": "str"},
        )

        documents = [
            Document(
                content=row["Description"],
                id=row["Code"],
                meta={"code": row["Code"]},
            )
            for i, row in cpc_list.iterrows()
        ]

    # Set how to embed the documents depending on provider
    if provider == "openai":
        embedding_function = "OpenAIEmbeddingFunction"
    else:
        embedding_function = "HuggingFaceEmbeddingFunction"

    # Initialise ChromaDocumentStore
    doc_store = MyChromaDocumentStore(
        persist_path=doc_store_dir,
        embedding_function=embedding_function,
        api_key=os.environ[env_key],
        model_name=model_name,
    )

    # Write documents if creating for the first time
    if documents:
        doc_store.write_documents(documents)

    return doc_store


def get_query_filters(
    ledger: pd.DataFrame, isic_codes: List[str]
) -> Dict[str, List[str]]:
    """Get query filters based on the ISIC codes in Haystack filter syntax,
    which means that metadata 'isic' has to be in the given list of ISIC codes

    Args:
        isic_codes (List[str]): List of ISIC codes

    Returns:
        Dict[str, Dict[str, List[str]]]: _description_
    """

    cpc_codes = ledger[ledger.isic_code.isin(isic_codes)]["cpc_code"].tolist()

    if len(cpc_codes) == 0:
        return {}

    return {"code": cpc_codes}


def get_ledger_isic(ledger: pd.DataFrame, cpc_code: str) -> List[str]:
    isic_codes = ledger[ledger.cpc_code == cpc_code]["isic_code"].tolist()

    return isic_codes


def get_ledger(res_dir: str):
    ledger = pd.read_csv(
        f"{res_dir}/20240626_LedgerV32_PositiveOnly.txt",
        delimiter="|",
        dtype={"cpc_code": str, "isic_code": str},
    )

    return ledger


def initialse(provider: str, res_dir: str, doc_store_dir: str):
    provider = provider
    # Initialise embedders to embed incoming queries for retrieval
    if provider == "openai":
        model_name = "text-embedding-3-small"
        env_key = "OPENAI_API_KEY"

    else:
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        env_key = "HUGGINGFACEHUB_API_TOKEN"

    # Initialise / Load document store with CPC descriptions for retrieval
    doc_store = get_document_store(
        provider, res_dir, doc_store_dir, env_key, model_name
    )

    ledger = get_ledger(res_dir)

    return doc_store, ledger


def save_results(results, counter):
    write_json(f"cpc_partial_results_{counter}.json", results)


def retrieve(
    embedded_docs: List[Dict[str, Any]], isic_results, doc_store, ledger, top_k
) -> Dict[str, Dict[str, List]]:
    """For each company, retrieve the top_k results for similar CPC codes

    Args:
        data (List[Dict[str, Any]]): List of company information

    Returns:
        List[Dict[str, List[Document]]]: List of companies with top_k CPC codes
    """

    results = {}

    for doc in tqdm(embedded_docs):
        company_id = doc.meta["company_id"]

        isic_codes = isic_results[company_id]["preds"]

        query_filter = get_query_filters(ledger, isic_codes)

        if query_filter == {}:
            # Get top_k results with the filter
            top_results = doc_store.search_embeddings(doc.embedding, top_k=top_k)[0]
        else:
            # Get top_k results with the filter
            top_results = doc_store.search_embeddings(
                doc.embedding, top_k=top_k, filters=query_filter
            )[0]

        # Structure the retrieval results neatly
        results[company_id] = {
            "preds": [
                [result.id, get_ledger_isic(ledger, result.id)]
                for result in top_results
            ],
            "scores": [result.score for result in top_results],
        }

    return results


def run(embedded_docs, isic_results, provider, res_dir, doc_store_dir, top_k):
    doc_store, ledger = initialse(provider, res_dir, doc_store_dir)

    cpc_results = retrieve(embedded_docs, isic_results, doc_store, ledger, top_k)

    return cpc_results
