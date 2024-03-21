from preprocess.preprocess_tilt import run as run_tilt
from preprocess.preprocess_orbis import run as run_orbis
from preprocess.preprocess_ecoinvent import run as run_ecoinvent

from preprocess.create_ecoinvent_dataset import (
    create_orbis_ecoinvent_dataset,
    create_ecoinvent_data_orbis,
)

from utils import load_env_file

import argparse


def run_preprocessing_pipeline(orbis_data_dir, tilt_data_dir, res_dir, save_dir):

    run_tilt(tilt_data_dir, res_dir, save_dir)
    run_orbis(orbis_data_dir, res_dir, save_dir)
    run_ecoinvent(res_dir, save_dir)


def run_dataset_pipeline(data_dir, save_dir):

    df_train_input, df_train_labels, df_val_input, df_val_labels, df_test = (
        create_orbis_ecoinvent_dataset(data_dir)
    )

    create_ecoinvent_data_orbis(
        df_train_input, df_train_labels, df_val_input, df_val_labels, df_test, save_dir
    )

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orbis_data_dir", type=str, default="./data/input/orbis")
    parser.add_argument(
        "--tilt_data_dir", type=str, default="./data/input/tiltData-v1.1.0"
    )
    parser.add_argument("--res_dir", type=str, default="./data/resources")
    parser.add_argument("--processed_dir", type=str, default="./data/processed")
    parser.add_argument("--dataset_dir", type=str, default="./data/dataset")

    args = parser.parse_args()

    load_env_file()

    run_preprocessing_pipeline(
        args.orbis_data_dir, args.tilt_data_dir, args.res_dir, args.processed_dir
    )

    run_dataset_pipeline(args.processed_dir, args.dataset_dir)
