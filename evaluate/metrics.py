from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight


def calculate_accuracy_numeric(labels: list, preds: list) -> float:
    sample_weights = compute_sample_weight(class_weight="balanced", y=labels)

    # return accuracy_score(labels, preds, sample_weight=sample_weights)
    return accuracy_score(labels, preds)


def calculate_accuracy_textual(labels: list, preds: list, multi_label: bool) -> float:

    num_samples = len(preds)

    if multi_label:
        correct = [1 for i in range(num_samples) if preds[i] in labels[i]]
    else:
        correct = [1 for i in range(num_samples) if labels[i] == preds[i]]

    # return accuracy_score(labels, preds, sample_weight=sample_weights)
    return sum(correct) / num_samples


def calculate_f1(labels: list, preds: list) -> tuple[float, float, float]:
    sample_weights = compute_sample_weight(class_weight="balanced", y=labels)
    average = "weighted"
    precision = precision_score(
        labels, preds, average=average, sample_weight=sample_weights
    )
    recall = recall_score(labels, preds, average=average, sample_weight=sample_weights)
    f1 = f1_score(labels, preds, average=average, sample_weight=sample_weights)
    return f1, precision, recall


def calculate_accuracy(
    labels: list, preds: list, multi_label: bool, numeric: bool = True
):

    if numeric:
        return calculate_accuracy_numeric(labels, preds)

    return calculate_accuracy_textual(labels, preds, multi_label=multi_label)
