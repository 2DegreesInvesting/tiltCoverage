from haystack import Document
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever

from typing import Optional, List

import pandas as pd

import os
import pickle


class EcoinventRetriever:
    def __init__(
        self,
        res_dir,
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
                self.embedder_model = "sentence-transformers/all-MiniLM-L6-v2"
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

        self.__get_document_store(res_dir)
        self.retriever = self.__get_retriever()
        if not self.load_doc_store:
            model_name = self.embedder_model.split("/")[-1]

            doc_store_path = (
                f"{res_dir}/document_store_{self.provider}_{model_name}.index"
            )
            doc_store_config_path = (
                f"{res_dir}/document_store_{self.provider}_{model_name}.json"
            )
            self.doc_store.save(doc_store_path, doc_store_config_path)

    def __get_document_store(self, doc_store_dir):
        model_name = self.embedder_model.split("/")[-1]

        doc_store_path = (
            f"{doc_store_dir}/document_store_{self.provider}_{model_name}.index"
        )
        doc_store_config_path = (
            f"{doc_store_dir}/document_store_{self.provider}_{model_name}.json"
        )

        if os.path.isfile(doc_store_path):
            self.load_doc_store = True
            self.doc_store = FAISSDocumentStore.load(
                doc_store_path, doc_store_config_path
            )

        else:
            self.load_doc_store = False
            self.doc_store = FAISSDocumentStore(
                sql_url=f"sqlite:///{doc_store_dir}/document_store_{self.provider}_{model_name}.db",
                duplicate_documents="skip",
                similarity="cosine",
                embedding_dim=self.embedding_size,
                return_embedding=True,
            )
            self.__init_doc_store(doc_store_dir)

    def __get_data(self, res_dir):
        ecoinvent = pd.read_csv(f"./data/resources/ecoinvent_complete.csv").rename(
            columns={
                "Reference Product Name": "ei_product",
                "Activity Name": "ei_activity",
                "Activity UUID & Product UUID": "activity_uuid_product_uuid",
            }
        )[["ei_product", "ei_activity", "activity_uuid_product_uuid"]]

        return ecoinvent

    def __init_doc_store(self, res_dir: str):
        data = self.__get_data(res_dir)

        documents = [
            Document(
                content=row["ei_activity"] + ";" + row["ei_product"],
                id=str(row["activity_uuid_product_uuid"]),
                meta={"id": str(row["activity_uuid_product_uuid"])},
            )
            for _, row in data.iterrows()
        ]

        self.doc_store.write_documents(documents, batch_size=self.batch_size)

    def __get_retriever(self):
        retriever = EmbeddingRetriever(
            embedding_model=self.embedder_model,
            document_store=self.doc_store,
            use_gpu=False,
            top_k=self.top_k,
            api_key=self.api_key,
            openai_organization=self.organization,
            batch_size=self.batch_size,
        )
        if not self.load_doc_store:
            retriever.embed_documents(self.doc_store)
            self.doc_store.update_embeddings(retriever)

        return retriever

    def retrieve(self, query: List[str]) -> List[Document]:

        results = self.retriever.retrieve_batch(query, batch_size=self.batch_size)

        return [query_result[0] for query_result in results]

    def prep_input(self, x):
        data = [self.prep_input_fnc(item) for item in x]

        return data

    def description_exists(self, obj):
        if obj and not isinstance(obj, float):
            return True

        return False

    def predict(self, batch):

        data = self.prep_input(batch)

        results = self.retrieve(data)
        preds = [r.id for r in results]

        assert len(preds) == len(data), "Input and output size do not match"

        return preds


def get_company_info(info_dict):

    naics = info_dict["naics_core_descr"]
    nace = info_dict["nace_core_descr"]
    ps = info_dict["products_and_services"]
    trade_description = info_dict["trade_description"]

    info = [naics, nace]

    # if trade_description and not isinstance(trade_description, float):
    #     info.append(trade_description)

    if ps and not isinstance(ps, float):
        info.append(ps)

    return "; ".join(info)
