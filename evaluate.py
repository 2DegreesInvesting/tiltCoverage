import argparse

from evaluate.evaluate_ecoinvent import evaluate_ecoinvent_method
from generate.ecoinvent_retrieval import predict_ecoinvent, prep_orbis_data

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/dataset")
    parser.add_argument("--other", type=str, default="./data/processed")

    args = parser.parse_args()

    predict = lambda x: predict_ecoinvent(x, data_dir=args.data_dir)

    evaluate_ecoinvent_method(
        "haystack_embedding_retrieval",
        prep_orbis_data,
        predict,
        args.data_dir,
    )
