import argparse

from src.utils import load_env_file

from src.preprocess import run_preprocessing
from src.mapping import run_ledger_mapping
from src.evaluate import run_manual_inspection


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./data/input/companyinfo")
    parser.add_argument("--data_dir", type=str, default="./data/dataset")
    parser.add_argument("--save_dir", type=str, default="./data/output")
    parser.add_argument("--res_dir", type=str, default="./data/resources")
    parser.add_argument("--doc_store_dir", type=str, default="./data/my_doc_store")

    parser.add_argument(
        "--provider", type=str, default="openai", choices=["openai", "hf"]
    )

    load_env_file()

    args = parser.parse_args()

    # run_preprocessing(input_dir=args.input_dir, save_dir=args.data_dir)

    # run_ledger_mapping(
    #     args.provider,
    #     args.data_dir,
    #     args.res_dir,
    #     args.doc_store_dir,
    #     args.save_dir,
    # )

    run_manual_inspection(args.data_dir, args.res_dir, args.save_dir)
