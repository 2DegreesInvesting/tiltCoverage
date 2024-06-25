from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.embedders import HuggingFaceAPIDocumentEmbedder
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.utils import Secret
from haystack import Document

from typing import List, Any, Dict

import pandas as pd
import os
import time



class TiltISICRetriever:
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

        # Initialise / Load document store with ISIC descriptions for retrieval
        self.doc_store = self.__get_doc_store(res_dir, doc_store_dir)

    def __get_doc_store(self, res_dir: str, doc_store_dir: str) -> ChromaDocumentStore:
        """_summary_

        Args:
            res_dir (str): Directory to find the ISIC table
            doc_store_dir (str): Directory where to save/load doc store

        Returns:
            ChromaDocumentStore: DocumentStore of ISIC code and descriptions
        """

        doc_store_dir = f"{doc_store_dir}/isic/{self.provider}"
        doc_store = f"{doc_store_dir}/chroma.sqlite3"

        # If doc_store file exists, then no need to write the documents
        if os.path.isfile(doc_store):
            print(">Load existing ISIC doc store")
            documents = None

        # Else, create documents out of each ISIC code and description
        else:
            print("> Create ISIC doc store")
            isic_list = pd.read_csv(
                f"{res_dir}/ISIC_Rev_4_english_structure.Txt", dtype={"Code": "str"}
            )

            documents = [
                Document(
                    content=row["Description"],
                    id=row["Code"],
                    meta={"isic": row["Code"][:2]},
                )
                for i, row in isic_list.iterrows()
                if len(row["Code"]) >= 2
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

    def __get_query_filters(self, isic_section: str) -> Dict[str, str]:
        """Get query filters based on the ISIC section groups in Haystack filter syntax,
        which means that metadata 'isic' has to equal the ISIC section

        Args:
            isic_section (str): ISIC section

        Returns:
            Dict[str, str]: ISIC filter statement according to haystack syntax
        """
        # TODO: needs to change to accept multiple ISIC sections for multiple SBIs
        return {"isic": isic_section}

    def __group_by_isic_section(
        self,
        companies: pd.DataFrame,
        sbi_activities: pd.DataFrame,
        companies_sbi_activities: pd.DataFrame,
    ) -> Dict[str, List[Any]]:
        """Transform company information into query documents for retrieval and
        group by the ISIC sections based on the first two digits of the SBI code

        Args:
            companies (pd.DataFrame): Companies table
            sbi_activities (pd.DataFrame): SBI activities table
            companies_sbi_activities (pd.DataFrame): Companies - SBI activities mapper tablelist_company_info (List[Dict[str, Any]]): List of company info dictionaries

        Returns:
            Dict[str, List[Any]]: Dictionary of ISIC section codes and list of company query documents
        """

        # TODO: needs to change to accept multiple ISIC sections for multiple SBIs
        section_groups = {}

        companies = companies[["company_id", "company_description"]]
        companies = companies.merge(companies_sbi_activities, on="company_id")
        companies = companies.merge(sbi_activities, on="sbi_code")
        companies["isic_section"] = companies.sbi_code.apply(lambda x: x[:2])
        companies["query"] = companies.apply(lambda x: f"{x["sbi_code_description"]}; {x["company_description"] or ""}", axis=1)

        for i, company in companies.iterrows():

            # Make query documents out of company information
            query_document = Document(
                content=company["query"],
                meta={"company_id": company["company_id"]},
            )

            query_section = company["isic_section"]

            if query_section in section_groups:
                section_groups[query_section].append(query_document)
            else:
                section_groups[query_section] = [query_document]

        return section_groups

    def __structure_results(
        self, query_docs: List[Document], retrieval_results: List[List[Document]]
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
            query_docs[i].meta["company_id"]: [
                result.id for result in retrieval_results[i]
            ]
            for i in range(n_docs)
        }

        return results

    def retrieve(
        self,
        companies: pd.DataFrame,
        sbi_activities: pd.DataFrame,
        companies_sbi_activities: pd.DataFrame,
    ) -> Dict[str, List[Document]]:
        """For each company, retrieve the top_k results for similar ISIC codes

        Args:
            companies (pd.DataFrame): Companies table
            sbi_activities (pd.DataFrame): SBI activities table
            companies_sbi_activities (pd.DataFrame): Companies - SBI activities mapper table

        Returns:
            Dict[str, List[Document]]: List of companies with top_k ISIC codes
        """

        # Group the data by ISIC sections
        data_isic_groups = self.__group_by_isic_section(
            companies, sbi_activities, companies_sbi_activities
        )

        results = {}
        docs = []
        for isic_section in data_isic_groups:

            query_docs = data_isic_groups[isic_section]
            query_filter = self.__get_query_filters(isic_section)

            # Get embeddings of the query documents
            embedded_docs = [doc for doc in self.embedder.run(query_docs)["documents"]]
            query_embedding = [doc.embedding for doc in embedded_docs]

            # TODO: where to choose similarity function?
            # TODO: exception when there are no results returned
            # Get top_k results with the filter
            top_results = self.doc_store.search_embeddings(
                query_embedding, top_k=self.top_k, filters=query_filter
            )

            # Structure the retrieval results neatly
            structured_results = self.__structure_results(query_docs, top_results)
            results.update(structured_results)
            docs.extend(embedded_docs)

            # Sleep to avoid reaching rate limit
            time.sleep(3)

        return docs, results

    def pretty_output(self, results: List[List[Document]]) -> Dict[str, List[List]]:

        # Get the preds and scores into a dictionary
        output = {
            "preds": [
                [isic.id for isic in results[company_id]] for company_id in results
            ],
            "scores": [
                [isic.score for isic in results[company_id]] for company_id in results
            ],
        }

        return output
