from haystack import Document
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever

from huggingface_hub import InferenceClient

from openai import OpenAI

from typing import Optional, List, Any, Dict

import pandas as pd
import numpy as np

import utils
import os


class LedgerMapper:
    def __init__(self, res_dir, output_dir):
        self.isic_retriever = ISICRetriever(res_dir, "openai")
        self.cpc_retriever = CPCProductRetriever(res_dir, "hf")
        self.activity_retriever = ActivityRetriever("hf")
        self.geo_retrieiver = GeoRetriever()

        self.res_dir = res_dir

        self.output_dir = output_dir

    def __structure_isic_output(self, res_dir, preds, scores):
        isic = pd.read_csv(
            f"{res_dir}/ISIC_Rev_4_english_structure.Txt", dtype={"Code": "str"}
        )
        isic.drop_duplicates(inplace=True)
        isic = isic.set_index("Code")
        isic = isic.to_dict()["Description"]

        preds_descr = [": ".join([p, isic.get(p, "???")]) for p in preds]

        output = [[preds_descr[i], scores[i]] for i in range(len(preds))]
        return output

    def __structure_cpc_output(self, res_dir, preds, scores):
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

    def structure_output(self, x, isic, cpc, activity, geo):
        output = []
        for i in range(len(x)):
            output.append(
                {
                    "input": x[i],
                    "output": {
                        "isic": isic[i],
                        "cpc": cpc[i],
                        "activity": activity[i],
                        "geo": geo[i],
                    },
                }
            )

        return output

    def map_to_ledger(self, x):
        print(">> ISIC")
        isic_preds, isic_scores = self.isic_retriever.predict(x)
        print(">> CPC")

        cpc_preds, cpc_scores = self.cpc_retriever.predict(x)
        print(">> Activity")
        act_preds = self.activity_retriever.predict(x, isic_preds)
        print(">> Geo")
        geo_preds = self.geo_retrieiver.predict(x)

        isic_results = self.__structure_isic_output(
            self.res_dir, isic_preds, isic_scores
        )
        cpc_results = self.__structure_cpc_output(self.res_dir, cpc_preds, cpc_scores)

        self.output = self.structure_output(
            x, isic_results, cpc_results, act_preds, geo_preds
        )

    def save_to_file(self):
        filename = f"{self.output_dir}/ci_output.json"

        utils.write_json(filename, self.output)

        return filename


class ISICRetriever:
    def __init__(self, res_dir, provider, embedder_model=None):
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

        self.__get_document_store(res_dir)
        self.retriever = self.__get_retriever()
        if not self.load_doc_store:

            doc_store_path = f"{res_dir}/isic_{self.provider}.index"
            doc_store_config_path = f"{res_dir}/isic_{self.provider}.json"
            self.doc_store.save(doc_store_path, doc_store_config_path)

    def __get_document_store(self, doc_store_dir):
        doc_store_path = f"{doc_store_dir}/isic_{self.provider}.index"
        doc_store_config_path = f"{doc_store_dir}/isic_{self.provider}.json"

        if os.path.isfile(doc_store_path):
            self.load_doc_store = True
            self.doc_store = FAISSDocumentStore.load(
                doc_store_path, doc_store_config_path
            )

        else:
            self.load_doc_store = False
            self.doc_store = FAISSDocumentStore(
                sql_url=f"sqlite:///{doc_store_dir}/isic_{self.provider}.db",
                similarity="cosine",
                embedding_dim=self.embedding_size,
                return_embedding=True,
            )
            self.__init_doc_store(doc_store_dir)

    def __init_doc_store(self, res_dir: str):
        cpc_list = pd.read_csv(
            f"{res_dir}/ISIC_Rev_4_english_structure.Txt", dtype={"Code": "str"}
        )

        documents = [
            Document(
                content=row["Description"],
                id=row["Code"],
                meta={"isic": row["Code"][:2]},
            )
            for i, row in cpc_list.iterrows()
        ]

        print(len(documents), "documents in the doc store")

        self.doc_store.write_documents(documents, batch_size=self.batch_size)

    def __get_retriever(self):
        retriever = EmbeddingRetriever(
            embedding_model=self.embedder_model,
            document_store=self.doc_store,
            use_gpu=False,
            # top_k=self.top_k,
            api_key=os.environ["OPENAI_API_KEY"],
            openai_organization=os.environ["OPENAI_ORG"],
            batch_size=self.batch_size,
        )
        if not self.load_doc_store:
            retriever.embed_documents(self.doc_store)
            self.doc_store.update_embeddings(retriever)

        return retriever

    def retrieve(self, query: List[str], filters) -> List[Document]:

        results = self.retriever.retrieve_batch(
            query, batch_size=self.batch_size, top_k=2498
        )

        results = self.filter_results(results, filters)

        return results

    def filter_results(self, results, filters):
        filtered_results = []
        n = len(results)

        for i in range(n):
            query_result = results[i]
            query_filter = filters[i]
            filtered_query_result = None

            for res in query_result:
                if res.meta["isic"] == query_filter:
                    filtered_query_result = res
                    break

            if not filtered_query_result:
                raise RuntimeError

            filtered_results.append(filtered_query_result)

        return filtered_results

    def get_isic_filters(self, item):
        return item["isic_code"][:2]

    def prep_input_fnc(self, info_dict):
        return info_dict["sbi"]

    def prep_input(self, x):
        data = [self.prep_input_fnc(item) for item in x]

        filters = [self.get_isic_filters(item) for item in x]

        return data, filters

    def predict(self, batch):

        data, filters = self.prep_input(batch)
        results = self.retrieve(data, filters)
        preds = [r.id for r in results]
        scores = [r.score for r in results]

        assert len(preds) == len(data), "Input and output size do not match"

        return preds, scores


class GeoRetriever:
    def __init__(self):
        pass

    def predict(self, batch):
        return ["nl"] * len(batch)


class ActivityRetriever:

    def __init__(self, provider, embedder_model=None):

        self.provider = provider
        self.batch_size = 4
        if not embedder_model:
            if provider == "openai":
                self.embedder_model = "text-embedding-3-small"
                self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

            else:
                self.embedder_model = "google/flan-t5-xxl"

                self.client = InferenceClient(
                    model=self.embedder_model,
                    token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
                )

        self.isic_act_mapper = self.__read_mapper()

        self.prompt_template = self.__read_prompt_template()

    def __read_mapper(self):
        mapper = utils.read_json("data/resources/isic_activity_mapper.json")

        return mapper

    def __read_prompt_template(self):
        with open("data/resources/activity_prompt.txt") as f:
            prompt_template = f.read()

        return prompt_template

    def __isic_in_mapper(self, isic_code):
        if isic_code in self.isic_act_mapper:
            return True, self.isic_act_mapper[isic_code]
        elif isic_code[:3] in self.isic_act_mapper:
            return True, self.isic_act_mapper[isic_code[:3]]
        elif isic_code[:2] in self.isic_act_mapper:
            return True, self.isic_act_mapper[isic_code[:2]]

        return False, []

    def __generate(self, prompt):
        if self.provider == "openai":
            response = self.client.completions.create(
                model=self.embedder_model, prompt=prompt
            )
            return response["choice"]
        else:
            response = self.client.text_generation(prompt)
            return response.choice

    def predict(self, batch, isic_preds):

        batch_activity = []

        for i, company in enumerate(batch):
            isic_code = isic_preds[i]

            isic_in_mapper, activity = self.__isic_in_mapper(isic_code)

            if isic_in_mapper:
                batch_activity.append(activity)
                continue

            if not company["sbi"]:
                print("huh?")

            if not company.get("description", ""):
                print("nuh huh")
            prompt_input = " ".join([company["sbi"], company.get("description", "")])
            prompt = self.prompt_template.format(input=prompt_input)
            output = self.__generate(prompt)

            # batch_activity.append("???")

            # # TODO accept multiple activities
            if output == "both":
                batch_activity.append(
                    ["ordinary transforming activity", "market activity"]
                )
            elif output == "transforming activity":
                batch_activity.append(["ordinary transforming activity"])
            else:
                batch_activity.append([output])

        return batch_activity


class CPCProductRetriever:
    def __init__(
        self,
        res_dir,
        provider,
        embedder_model=None,
    ):

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

        self.__get_document_store(res_dir)
        self.retriever = self.__get_retriever()
        if not self.load_doc_store:
            model_name = self.embedder_model.split("/")[-1]

            doc_store_path = f"{res_dir}/cpc_{self.provider}_{model_name}.index"
            doc_store_config_path = f"{res_dir}/cpc_{self.provider}_{model_name}.json"
            self.doc_store.save(doc_store_path, doc_store_config_path)

    def __get_document_store(self, doc_store_dir):
        model_name = self.embedder_model.split("/")[-1]

        doc_store_path = f"{doc_store_dir}/cpc_{self.provider}_{model_name}.index"
        doc_store_config_path = f"{doc_store_dir}/cpc_{self.provider}_{model_name}.json"

        if os.path.isfile(doc_store_path):
            self.load_doc_store = True
            self.doc_store = FAISSDocumentStore.load(
                doc_store_path, doc_store_config_path
            )

        else:
            self.load_doc_store = False
            self.doc_store = FAISSDocumentStore(
                sql_url=f"sqlite:///{doc_store_dir}/cpc_{self.provider}_{model_name}.db",
                similarity="cosine",
                embedding_dim=self.embedding_size,
                return_embedding=True,
            )
            self.__init_doc_store(doc_store_dir)

    def __init_doc_store(self, res_dir: str):
        cpc_list = pd.read_csv(f"{res_dir}/cpc_list.csv", dtype={"cpc": "str"})

        documents = [
            Document(
                content=row["description"], id=row["cpc"], meta={"isic": row["isic"]}
            )
            for i, row in cpc_list.iterrows()
        ]

        print(len(documents), "documents in the doc store")

        self.doc_store.write_documents(documents, batch_size=self.batch_size)

    def __get_retriever(self):
        retriever = EmbeddingRetriever(
            embedding_model=self.embedder_model,
            document_store=self.doc_store,
            use_gpu=False,
            # top_k=self.top_k,
            api_key=os.environ["OPENAI_API_KEY"],
            openai_organization=os.environ["OPENAI_ORG"],
            batch_size=self.batch_size,
        )
        if not self.load_doc_store:
            retriever.embed_documents(self.doc_store)
            self.doc_store.update_embeddings(retriever)

        return retriever

    def retrieve(self, query: List[str], filters) -> List[Document]:

        results = self.retriever.retrieve_batch(
            query, batch_size=self.batch_size, top_k=2498
        )

        results = [res[0] for res in results]
        # results = self.filter_results(results, filters)

        return results

    def filter_results(self, results, filters):
        filtered_results = []
        n = len(results)

        for i in range(n):
            query_result = results[i]
            query_filter = filters[i]
            filtered_query_result = None

            for res in query_result:
                if res.meta["isic"] in query_filter:
                    filtered_query_result = res

            if not filtered_query_result:
                raise RuntimeError

            filtered_results.append(filtered_query_result)

        return filtered_results

    def get_isic_filters(self, item):
        return [item["isic_code"][:2], "na"]

    def prep_input_fnc(self, info_dict):
        details = [
            "sbi",
            "description",
        ]

        info = []
        for key in details:
            descr = info_dict.get(key, None)

            if descr:
                info.append(descr)
        return "; ".join(info)

    def prep_input(self, x):
        data = [self.prep_input_fnc(item) for item in x]

        filters = [self.get_isic_filters(item) for item in x]

        return data, filters

    def description_exists(self, obj):
        if obj and not isinstance(obj, float):
            return True

        return False

    def predict(self, batch):

        data, filters = self.prep_input(batch)
        results = self.retrieve(data, filters)
        preds = [r.id for r in results]
        scores = [r.score for r in results]

        assert len(preds) == len(data), "Input and output size do not match"

        return preds, scores
