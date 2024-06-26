from haystack import Document
from haystack.document_stores import FAISSDocumentStore, InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever

from typing import Optional, List

import pandas as pd

import os
import pickle


class EcoinventStepWiseRetriever:
    def __init__(
        self,
        res_dir,
        save_dir,
        provider,
        top_k,
        prep_input_fnc,
        embedder_model=None,
        organization=None,
        api_key=None,
        load_doc_store=False,
    ):
        assert provider in ["openai", "hf"], f"Unknown provider: {provider}"

        self.provider = provider
        self.batch_size = 32

        if not embedder_model:
            if provider == "openai":
                self.embedder_model = "text-embedding-3-small"
            else:
                self.embedder_model = (
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
        else:
            self.embedder_model = embedder_model

        print(f"Initialise {self.embedder_model}")

        if provider == "openai":
            if self.embedder_model == "text-embedding-3-small":

                self.embedding_size = 1536  # TODO: could change it to a different size as input to the API call
        else:
            self.embedding_size = 384

        self.organization = organization
        self.api_key = api_key

        self.prep_input_fnc = prep_input_fnc
        self.top_k = top_k

        self.load_doc_Store = load_doc_store

        self.__get_document_store(save_dir)
        self.activity_retriever, self.product_retriever = self.__get_retriever()

    def __get_document_store(self, doc_store_dir):
        model_name = self.embedder_model.split("/")[-1]

        ds_type_list = ["activity", "product"]

        for i, ds_type in enumerate(ds_type_list):
            self.load_doc_store = False

            doc_store = InMemoryDocumentStore(
                duplicate_documents="skip",
                similarity="cosine",
                embedding_dim=self.embedding_size,
                return_embedding=True,
            )

            self.__setattr__(f"{ds_type}_doc_store", doc_store)

        if not self.load_doc_store:

            self.__init_doc_store(doc_store_dir)

    def __get_data(self, res_dir):
        ecoinvent = pd.read_csv(
            f"./data/resources/ecoinvent_complete_no_geo.csv"
        ).rename(
            columns={
                "Reference Product Name": "ei_product",
                "Activity Name": "ei_activity",
            }
        )[
            [
                "ei_product",
                "ei_activity",
                "ei_product_id",
                "ei_activity_id",
                "activity_id_product_id",
            ]
        ]

        return ecoinvent

    def __init_doc_store(self, res_dir: str):
        data = self.__get_data(res_dir)

        activity_data = data.drop_duplicates(subset="ei_activity_id")
        activity_doc = [
            Document(
                content=row["ei_activity"],
                id=str(row["ei_activity_id"]),
            )
            for _, row in activity_data.iterrows()
        ]

        product_doc = [
            Document(
                content=row["ei_product"],
                id=str(row["activity_id_product_id"]),
                meta={
                    "ei_activity_id": row["ei_activity_id"],
                    "activity_id_product_id": row["activity_id_product_id"],
                },
            )
            for _, row in data.iterrows()
        ]

        self.activity_doc_store.write_documents(
            activity_doc, batch_size=self.batch_size
        )
        self.product_doc_store.write_documents(product_doc, batch_size=self.batch_size)

    def __get_retriever(self):
        activity_retriever = EmbeddingRetriever(
            embedding_model=self.embedder_model,
            document_store=self.activity_doc_store,
            use_gpu=False,
            top_k=self.top_k,
            api_key=self.api_key,
            openai_organization=self.organization,
            batch_size=self.batch_size,
        )
        if not self.load_doc_store:
            activity_retriever.embed_documents(self.activity_doc_store)
            self.activity_doc_store.update_embeddings(activity_retriever)

        product_retriever = EmbeddingRetriever(
            embedding_model=self.embedder_model,
            document_store=self.product_doc_store,
            use_gpu=False,
            top_k=self.top_k,
            api_key=self.api_key,
            openai_organization=self.organization,
            batch_size=self.batch_size,
        )
        if not self.load_doc_store:
            product_retriever.embed_documents(self.product_doc_store)
            self.product_doc_store.update_embeddings(product_retriever)

        return activity_retriever, product_retriever

    def retrieve_activity(self, query: List[str]) -> List[Document]:

        results = self.activity_retriever.retrieve_batch(
            query, batch_size=self.batch_size, top_k=1
        )

        return [query_result[0] for query_result in results]

    def retrieve_product(
        self, query: List[str], activity_filter: List[dict]
    ) -> List[Document]:

        print("batch work")
        results = self.product_retriever.retrieve_batch(
            query, batch_size=self.batch_size, filters=activity_filter, top_k=1
        )

        return [
            query_result[0] if len(query_result) > 0 else None
            for query_result in results
        ]

    def prep_input(self, x):
        data = [self.prep_input_fnc(item) for item in x]

        return data

    def description_exists(self, obj):
        if obj and not isinstance(obj, float):
            return True

        return False

    def predict(self, batch):

        data = self.prep_input(batch)

        print("activity_results")
        activity_results = self.retrieve_activity(data)
        print("create filter")

        filters = [{"ei_activity_id": {"$eq": r.id}} for r in activity_results]
        print("product_results")

        product_results = self.retrieve_product(data, filters)
        preds = [r.meta["activity_id_product_id"] if r else "" for r in product_results]
        assert len(preds) == len(data), "Input and output size do not match"

        return preds
