from dataclasses import dataclass, field
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score, f1_score
)
import pandas as pd
from evaluation.pipeline import Results

@dataclass(frozen=True)
class Approach:
    key: str
    file_path: str
    display_name: str
    
@dataclass
class Metric:
    name: str
    values: dict = field(default_factory=dict)

def get_results_from_metadata(file_path: str) -> Results:
    df = pd.read_csv(file_path, sep=';')
    y_true = df['true_label'].values
    y_pred = df['pred_label'].values
    return Results(y_true=y_true, y_pred=y_pred)

def get_metrics_for_multiclass(approaches: list[Approach]) -> list[Metric]:
    f1 = Metric('f1_score')
    accuracy = Metric('accuracy')
    precision = Metric('precision')
    recall = Metric('recall')
    f1_macro = Metric('f1_macro')
    for approach in approaches:
        results = get_results_from_metadata(approach.file_path)
        f1.values[approach] = f1_score(results.y_true, results.y_pred, average=None, zero_division=0)
        accuracy.values[approach] = accuracy_score(results.y_true, results.y_pred)
        precision.values[approach] = precision_score(results.y_true, results.y_pred, average=None, zero_division=0)
        recall.values[approach] = recall_score(results.y_true, results.y_pred, average=None, zero_division=0)
        f1_macro.values[approach] = f1_score(results.y_true, results.y_pred, average='macro', zero_division=0)
    
    return [f1, accuracy, precision, recall, f1_macro] 

def get_metrics_for_binary(approaches: list[Approach]) -> list[Metric]:
    f1 = Metric('f1_score')
    accuracy = Metric('accuracy')
    precision = Metric('precision')
    recall = Metric('recall')
    for approach in approaches:
        results = get_results_from_metadata(approach.file_path)
        f1.values[approach] = f1_score(results.y_true, results.y_pred, average='binary', zero_division=0)
        accuracy.values[approach] = accuracy_score(results.y_true, results.y_pred)
        precision.values[approach] = precision_score(results.y_true, results.y_pred, average='binary', zero_division=0)
        recall.values[approach] = recall_score(results.y_true, results.y_pred, average='binary', zero_division=0)
        
    return [f1, accuracy, precision, recall]

def generate_confusion_matrix(approach: Approach):
    results = get_results_from_metadata(approach.file_path)
    mat = confusion_matrix(results.y_true, results.y_pred)
    display = ConfusionMatrixDisplay(mat)
    return display
