from huggingface_hub import InferenceClient

from openai import OpenAI

from typing import List, Any, Dict

import pandas as pd
import os

from .. import utils


class TiltActivityRetriever:

    def __init__(self, provider, res_dir):

        self.provider = provider
        self.batch_size = 4
        self.res_dir = res_dir

        self.default = ["ordinary transforming activity", "market activity"]
        # if provider == "openai":
        #     self.embedder_model = "gpt-3.5-turbo-0125"
        #     self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        # else:
        #     self.embedder_model = "google/flan-t5-xxl"

        #     self.client = InferenceClient(
        #         model=self.embedder_model,
        #         token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        #     )

        self.isic_act_mapper = self.__read_mapper()

        self.prompt_template = self.__read_prompt_template()

    def __read_mapper(self):
        mapper = utils.read_json(f"{self.res_dir}/isic_activity_mapper.json")

        return mapper

    def __read_prompt_template(self):
        with open(f"{self.res_dir}/activity_prompt.txt") as f:
            prompt_template = f.read()

        return prompt_template

    def __generate(self, prompt):
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.embedder_model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        else:
            response = self.client.text_generation(prompt)
            return response.choice

    def retrieve(self, documents, isic_results):

        batch_activity = {}

        for company in documents:
            company_id = company.meta["company_id"]
            isic_codes = isic_results[company_id]
            activities = set()

            for code in isic_codes:
                code_act = self.isic_act_mapper.get(code, self.default)
                activities.union(code_act)
                if len(activities) > 1:
                    break

            batch_activity[company_id] = list(activities)

        return batch_activity
