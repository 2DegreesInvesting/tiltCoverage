from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from haystack.components.embedders import HuggingFaceAPIDocumentEmbedder
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.utils import Secret
from haystack import Document

from typing import List, Any, Dict

import pandas as pd
import os
import time


class TiltCPCRetriever:
    def __init__(
        self, provider: str, res_dir: str, doc_store_dir: str, top_k: int
    ) -> None:

        self.provider = provider
        self.top_k = top_k

        # Initialise embedders to embed incoming queries for retrieval
        if provider == "openai":
            self.model_name = "text-embedding-3-small"
            self.env_key = "OPENAI_API_KEY"

            self.embedder = OpenAIDocumentEmbedder(
                api_key=Secret.from_env_var(self.env_key),
                model=self.model_name,
            )
        else:
            self.model_name = (
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            self.env_key = "HUGGINGFACEHUB_API_TOKEN"

            self.embedder = HuggingFaceAPIDocumentEmbedder(
                api_type="serverless_inference_api",
                api_params={"model": self.model_name},
                token=Secret.from_env_var(self.env_key),
            )

        # Initialise / Load document store with CPC descriptions for retrieval
        self.doc_store = self.__get_doc_store(res_dir, doc_store_dir)

    def __get_doc_store(self, res_dir: str, doc_store_dir: str) -> ChromaDocumentStore:
        """Create a new or load existing document store with CPC code and descriptions

        Args:
            res_dir (str): Directory to find the CPC table
            doc_store_dir (str): Directory where to save/load doc store

        Returns:
            ChromaDocumentStore: DocumentStore of CPC code and descriptions
        """

        doc_store_dir = f"{doc_store_dir}/cpc/{self.provider}"
        doc_store = f"{doc_store_dir}/chroma.sqlite3"

        # If doc_store file exists, then no need to write the documents
        if os.path.isfile(doc_store):
            print(">Load existing CPC doc store")
            documents = None

        # Else, create documents out of each CPC code and description
        else:
            print("> Create CPC doc store")
            cpc_list = pd.read_csv(
                f"{res_dir}/cpc_list.csv", dtype={"cpc": "str", "isic": "str"}
            )

            documents = [
                Document(
                    content=row["description"],
                    id=row["cpc"],
                    meta={"isic": row["isic"]},
                )
                for i, row in cpc_list.iterrows()
            ]

        # Set how to embed the documents depending on provider
        if self.provider == "openai":
            self.embedding_function = "OpenAIEmbeddingFunction"
        else:
            self.embedding_function = "HuggingFaceEmbeddingFunction"

        # Initialise ChromaDocumentStore
        doc_store = ChromaDocumentStore(
            persist_path=doc_store_dir,
            embedding_function=self.embedding_function,
            api_key=os.environ[self.env_key],
            model_name=self.model_name,
        )

        # Write documents if creating for the first time
        if documents:
            doc_store.write_documents(documents)

        return doc_store

    def __get_query_filters(self, isic_codes: List[str]) -> Dict[str, List[str]]:
        """Get query filters based on the ISIC codes in Haystack filter syntax,
        which means that metadata 'isic' has to be in the given list of ISIC codes

        Args:
            isic_codes (List[str]): List of ISIC codes

        Returns:
            Dict[str, Dict[str, List[str]]]: _description_
        """

        return {"isic": isic_codes + ["na"]}

    def retrieve(
        self, embedded_docs: List[Dict[str, Any]], isic_results
    ) -> Dict[str, List[Document]]:
        """For each company, retrieve the top_k results for similar CPC codes

        Args:
            data (List[Dict[str, Any]]): List of company information

        Returns:
            List[Dict[str, List[Document]]]: List of companies with top_k CPC codes
        """

        results = {}
        for doc in embedded_docs:
            company_id = doc.meta["company_id"]

            isic_codes = isic_results[company_id]

            query_filter = self.__get_query_filters(isic_codes)

            # Get top_k results with the filter
            top_results = self.doc_store.search_embeddings(
                doc.embedding, top_k=self.top_k, filters=query_filter
            )

            # Structure the retrieval results neatly
            results[company_id] = [
                [result.id, result.meta["isic"]] for result in top_results[0]
            ]

            # Sleep to avoid reaching rate limit
            time.sleep(1)

        return results

    def pretty_output(self, results: List[List[Document]]) -> Dict[str, List[List]]:

        # Get the preds and scores into a dictionary
        output = {
            "preds": [
                [cpc.id for cpc in results[company_id]] for company_id in results
            ],
            "scores": [
                [cpc.score for cpc in results[company_id]] for company_id in results
            ],
        }

        return output
