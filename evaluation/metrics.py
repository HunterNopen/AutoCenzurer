from dataclasses import dataclass
from typing import NamedTuple, Sequence
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score
)
import pandas as pd
from evaluation.pipeline import Results


@dataclass
class Approach:
    """Configuration for a classification approach."""
    key: str
    file_path: str
    display_name: str


class MetricsResult(NamedTuple):
    """Calculated metrics for a classification approach."""
    precision: float
    recall: float
    accuracy: float


APPROACHES: list[Approach] = [
    Approach(
        key="llm_metadata",
        file_path="classification_inference_metadata_result.csv",
        display_name="LLM Metadata Classification",
    ),
    Approach(
        key="llm_metadata_v2",
        file_path="classification_inference_metadata_result.csv",
        display_name="LLM Metadata Classification 2",
    ),
]


def get_results_from_metadata(file_path: str) -> Results:
    """Load classification results from a CSV file."""
    df = pd.read_csv(file_path, sep=';')
    y_true = df['true_label'].values
    y_pred = df['pred_label'].values
    return Results(y_true=y_true, y_pred=y_pred)


def calculate_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> MetricsResult:
    """Calculate precision, recall, and accuracy for binary classification.

    Uses average='binary' and pos_label=1 explicitly for clarity.
    """
    precision = float(precision_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0))
    recall = float(recall_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    return MetricsResult(precision=precision, recall=recall, accuracy=accuracy)


def load_all_approaches() -> dict[str, Results]:
    """Load results for all approaches."""
    results: dict[str, Results] = {}
    for approach in APPROACHES:
        try:
            results[approach.key] = get_results_from_metadata(approach.file_path)
        except FileNotFoundError:
            print(f"Warning: Could not load {approach.file_path}")
    return results


def generate_confusion_matrix(y_true, y_pred):
    """Generate a confusion matrix display."""
    mat = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(mat)
    return display
