from pathlib import Path
from datasets import Dataset
from pandas import read_csv as pandas_read_csv
from typing import Iterator
from collections.abc import Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

SEED = 148 

@dataclass
class Batch:
    id: list[str]
    span_text: list[str]
    label: list[int]

@dataclass
class Results:
    y_true: list[int]
    y_pred: list[int]

def get_batched_dataset(data_path: Path, batch_size: int = 32) -> Dataset:
    df = pandas_read_csv(data_path, sep=';')
    dataset = Dataset.from_pandas(df)
    batches = dataset.shuffle(seed=SEED).batch(batch_size)

    return batches

def get_inference_results(
    data_iterator: Iterator[Dataset], 
    process_batch_fn: Callable[[Batch], Results]
    ) -> Results:
    aggregated_y_true = []
    aggregated_y_pred = []
    for batch in data_iterator:
        try:
            results = process_batch_fn(Batch(id=batch['id'], span_text=batch['span_text'], label=batch['label']))
        except:
            logging.exception("process batch error")
            continue
                    
        aggregated_y_true.extend(results.y_true)
        aggregated_y_pred.extend(results.y_pred)
        
    return Results(y_true=aggregated_y_true, y_pred=aggregated_y_pred)
