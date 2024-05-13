from haystack import Document
from haystack.document_stores import FAISSDocumentStore, InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever

from typing import Optional, List

import pandas as pd
import numpy as np

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
                self.embedder_model = (
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
        else:
            self.embedder_model = embedder_model
        print(f"Initialise {self.embedder_model}")

        if provider == "openai":
            if self.embedder_model == "text-embedding-3-small":

                self.embedding_size = 1536  # TODO: could change it to a different size as input to the API call

            elif self.embedder_model == "text-embedding-3-large":
                self.embedding_size = 3072
        else:
            if self.embedder_model.startswith("sentence-transformer"):

                self.embedding_size = 384
            else:
                self.embedding_size = 768

        self.organization = os.environ["OPENAI_ORG"]
        self.api_key = os.environ["OPENAI_API_KEY"]

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
                similarity="cosine",
                embedding_dim=self.embedding_size,
                return_embedding=True,
            )
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
            ["ei_product", "ei_activity", "activity_id_product_id"]
        ]

        return ecoinvent

    def __init_doc_store(self, res_dir: str):
        data = self.__get_data(res_dir)

        documents = [
            Document(
                content=row["ei_activity"] + ";" + row["ei_product"],
                id=str(row["activity_id_product_id"]),
                meta={"id": str(row["activity_id_product_id"])},
            )
            for _, row in data.iterrows()
        ]
        print(len(documents), "documents in the doc store")

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

    def __create_training_data(self, x, labels):
        data = self.prep_input(x)
        ecoinvent = (
            self.__get_data("").set_index("activity_id_product_id").to_dict("index")
        )

        def get_ecoinvent(aupu):
            return ecoinvent[aupu]

        training_data = [
            {"question": data[i], "pos_doc": get_ecoinvent(label)}
            for i in range(len(data))
            for label in labels[i]
        ]
        return training_data

    def train(self, x, labels):
        training_data = self.__create_training_data(x, labels)
        self.retriever.train(training_data=training_data, n_epochs=10)


def get_company_info_core_prim(info_dict):

    code_type = ["naics", "nace"]
    code_level = ["core", "prim"]

    info = []

    for ctype in code_type:
        for clevel in code_level:
            descr = info_dict.get(f"{ctype}_{clevel}_descr", None)
            if descr:
                info.append(descr)

    ps = info_dict["products_and_services"]
    trade_description = info_dict["trade_description"]

    # if trade_description and not isinstance(trade_description, float):
    #     info.append(trade_description)

    if ps and not isinstance(ps, float):
        info.append(ps)

    return "; ".join(info)


def get_company_info_sbi_core_prim(info_dict):

    code_type = ["naics", "nace", "sbi"]
    code_level = ["core", "prim"]

    info = []

    for ctype in code_type:
        for clevel in code_level:
            if ctype == "sbi" and clevel == "core":
                clevel = "sec"

            descr = info_dict.get(f"{ctype}_{clevel}_descr", None)

            if descr:
                info.append(descr)

    ps = info_dict["products_and_services"]
    trade_description = info_dict["trade_description"]

    # if trade_description and not isinstance(trade_description, float):
    #     info.append(trade_description)

    if ps and not isinstance(ps, float):
        info.append(ps)

    return "; ".join(info)


def get_company_info_core_prim_sec(info_dict):

    code_type = ["naics", "nace"]
    code_level = ["core", "prim", "sec"]

    info = []

    for ctype in code_type:
        for clevel in code_level:
            descr = info_dict.get(f"{ctype}_{clevel}_descr", None)
            if descr:
                info.append(descr)

    ps = info_dict["products_and_services"]
    trade_description = info_dict["trade_description"]

    # if trade_description and not isinstance(trade_description, float):
    #     info.append(trade_description)

    if ps and not isinstance(ps, float):
        info.append(ps)

    return "; ".join(info)


def get_company_info_sbi(info_dict):
    details = [
        "sbi_prim_descr",
        "sbi_sec_descr",
        # "trade_description",
        # "products_and_services",
    ]

    info = []
    for key in details:
        descr = info_dict.get(key, None)

        if descr:
            info.append(descr)
    return "; ".join(info)


class EcoinventBaselineRetriever:
    def __init__(self, res_dir, baseline_type):

        assert baseline_type in ["random", "majority"]
        self.baseline_type = baseline_type
        self.all_labels = self.__get_data(res_dir)

    def __get_data(self, res_dir: str):
        ecoinvent = pd.read_csv(f"{res_dir}/ecoinvent_complete_no_geo.csv")[
            "activity_id_product_id"
        ]

        return ecoinvent

    def random_baseline(self, batch):
        labels = self.all_labels.drop_duplicates().tolist()

        num_examples = len(batch)

        preds = np.random.choice(labels, size=num_examples)

        return preds

    def majority_baseline(self, batch):
        label = self.all_labels.mode().item()

        preds = [label] * len(batch)

        return preds

    def predict(self, batch):
        if self.baseline_type == "random":
            return self.random_baseline(batch)

        else:
            return self.majority_baseline(batch)
