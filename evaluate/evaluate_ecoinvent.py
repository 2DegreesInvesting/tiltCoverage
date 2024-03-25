from .metrics import calculate_accuracy
from .utils import read_labelled_data


def evaluate_ecoinvent_retriever_method(method_name, retriever, data_dir):

    print(f"Evaluating {method_name} on ecoinvent data")

    # read labelled data
    print(">Reading ecoinvent validation data")
    x, labels = read_labelled_data(data_dir, "val")

    # get prediction
    print(">Generating predictions")
    preds = retriever.predict(x)

    # get accuracy
    print(">Calculating prediction accuracy")
    accuracy = calculate_accuracy(labels, preds, multi_label=True, numeric=False)
    accuracy_percent = accuracy * 100

    print(f"{method_name} had {accuracy_percent:.2f}% accuracy")


def train_and_evaluate_ecoinvent_retriever_method(method_name, retriever, data_dir):

    print(f"Training {method_name} on ecoinvent data")

    # read labelled data
    print(">Reading ecoinvent training data")
    x, labels = read_labelled_data(data_dir, "train")

    # get prediction
    print(">Training retriever")
    retriever.train(x, labels)

    evaluate_ecoinvent_retriever_method(method_name, retriever, data_dir)
