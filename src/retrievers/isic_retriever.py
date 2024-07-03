from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from haystack import Document

from typing import List, Any, Dict

from ..utils import MyChromaDocumentStore, read_json, write_json

import pandas as pd
import os
import time
from tqdm import tqdm


def initialise(provider: str, res_dir: str, doc_store_dir: str) -> None:

    # Initialise embedders to embed incoming queries for retrieval
    if provider == "openai":
        model_name = "text-embedding-3-small"
        env_key = "OPENAI_API_KEY"

    else:
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        env_key = "HUGGINGFACEHUB_API_TOKEN"

    # Initialise / Load document store with ISIC descriptions for retrieval
    doc_store = get_doc_store(provider, res_dir, doc_store_dir, env_key, model_name)

    return doc_store


def get_doc_store(
    provider: str, res_dir: str, doc_store_dir: str, env_key, model_name
) -> ChromaDocumentStore:
    """_summary_

    Args:
        res_dir (str): Directory to find the ISIC table
        doc_store_dir (str): Directory where to save/load doc store

    Returns:
        ChromaDocumentStore: DocumentStore of ISIC code and descriptions
    """

    doc_store_dir = f"{doc_store_dir}/isic/{provider}"
    doc_store_path = f"{doc_store_dir}/chroma.sqlite3"

    # If doc_store file exists, then no need to write the documents
    if os.path.isfile(doc_store_path):
        print(">Load existing ISIC doc store")
        documents = None

    # Else, create documents out of each ISIC code and description
    else:
        print("> Create ISIC doc store")
        isic_list = pd.read_csv(f"{res_dir}/ISIC_rev_4.csv", dtype={"Code": "str"})

        documents = [
            Document(
                content=row["Description"],
                id=row["Code"],
                meta={"isic": row["Code"][:2]},
            )
            for i, row in isic_list.iterrows()
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


def get_query_filters(isic_section: str) -> Dict[str, str]:
    """Get query filters based on the ISIC section groups in Haystack filter syntax,
    which means that metadata 'isic' has to equal the ISIC section

    Args:
        isic_section (str): ISIC section

    Returns:
        Dict[str, str]: ISIC filter statement according to haystack syntax
    """
    # TODO: needs to change to accept multiple ISIC sections for multiple SBIs
    return {"isic": isic_section}


def group_by_isic_section(
    query_documents: List[Document],
) -> Dict[str, List[Document]]:
    """Transform company information into query documents for retrieval and
    group by the ISIC sections based on the first two digits of the SBI code
    """

    # TODO: needs to change to accept multiple ISIC sections for multiple SBIs
    section_groups = {}

    for doc in query_documents:

        query_section = doc.meta["isic_section"]

        if query_section in section_groups:
            section_groups[query_section].append(doc)
        else:
            section_groups[query_section] = [doc]

    return section_groups


def structure_results(
    query_docs: List[Document], retrieval_results: List[List[Document]]
) -> Dict[str, List[Document]]:
    """Structure the retrieval results into a dictionary of company_id and
    retrieved ISIC codes

    Args:
        query_docs (List[Document]): List of query documents
        retrieval_results (List[Document]): List of top_k ISIC codes retrieved for each query document

    Returns:
        Dict[str, List[Document]]: Dictionary of company_id and its top_k retrieved ISIC codes
    """

    n_docs = len(query_docs)

    results = {
        query_docs[i].meta["company_id"]: {
            "preds": [result.id for result in retrieval_results[i]],
            "scores": [result.score for result in retrieval_results[i]],
        }
        for i in range(n_docs)
    }

    return results


def save_results(results, section):
    write_json(f"isic_partial_results_{section}.json", results)


def retrieve(
    query_documents: List[Document], doc_store, top_k
) -> Dict[str, List[Document]]:

    # Group the data by ISIC sections
    data_isic_groups = group_by_isic_section(query_documents)

    results = {}
    for isic_section in data_isic_groups:

        if isic_section != "None" and int(isic_section) < 98:
            continue

        query_docs = data_isic_groups[isic_section]
        print(len(query_docs))
        all_query_embedding = [doc.embedding for doc in query_docs]

        step = 1024
        for i in tqdm(range(0, len(query_docs) + step, step)):
            query_embedding = all_query_embedding[i : i + step]
            # Get embeddings of the query documents

            # TODO: where to choose similarity function?
            # TODO: exception when there are no results returned
            # Get top_k results with the filter

            # TODO: every europages one results with NA
            if isic_section == "None":
                top_results = doc_store.search_embeddings(query_embedding, top_k=top_k)
            else:
                query_filter = get_query_filters(isic_section)
                top_results = doc_store.search_embeddings(
                    query_embedding, top_k=top_k, filters=query_filter
                )
            if len(top_results) == 0:
                continue
            # Structure the retrieval results neatly
            structured_results = structure_results(
                query_docs[i : i + step], top_results
            )
            results.update(structured_results)

    return results


def run(query_documents, provider, res_dir, doc_store_dir, top_k):
    doc_store = initialise(provider, res_dir, doc_store_dir)
    isic_results = retrieve(query_documents, doc_store, top_k)

    return isic_results
