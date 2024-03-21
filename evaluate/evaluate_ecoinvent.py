from .metrics import calculate_accuracy

import json


def read_labelled_data(data_dir: str):
    with open(f"{data_dir}/orbis_ecoinvent_val_dataset.json") as f:
        data = json.load(f)

    return data["x"], data["y"]


def evaluate_ecoinvent_method(method_name, predict, data_dir):

    print(f"Evaluating {method_name} on ecoinvent data")

    # read labelled data
    print(">Reading ecoinvent validation data")
    x, labels = read_labelled_data(data_dir)

    # get prediction
    print(">Generating predictions")
    preds = predict(x)

    # get accuracy
    print(">Calculating prediction accuracy")
    accuracy = calculate_accuracy(labels, preds, multi_label=True, numeric=False)
    accuracy_percent = accuracy * 100

    print(f"{method_name} had {accuracy_percent:.2f}% accuracy")
