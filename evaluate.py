import argparse

from utils import load_env_file

from evaluate.evaluate_ecoinvent import evaluate_ecoinvent_method
from generate.ecoinvent_retrieval import get_company_info, EcoinventRetriever

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/dataset")
    parser.add_argument("--other", type=str, default="./data/processed")

    load_env_file()

    args = parser.parse_args()

    retriever = EcoinventRetriever(
        args.other,
        "hf",
        1,
        get_company_info,
        embedder_model="google/flan-t5-xxl",
    )
    predict = lambda x: retriever.predict(x)

    evaluate_ecoinvent_method(
        "haystack_embedding_retrieval_hf",
        predict,
        args.data_dir,
    )

    # retriever = EcoinventRetriever(
    #     args.other,
    #     "openai",
    #     1,
    #     get_company_info,
    #     organization=os.environ["OPENAI_ORG"],
    #     api_key=os.environ["OPENAI_API_KEY"],
    # )

    # predict = lambda x: retriever.predict(x)

    # evaluate_ecoinvent_method(
    #     "haystack_embedding_retrieval_openai",
    #     predict,
    #     args.data_dir,
    # )
