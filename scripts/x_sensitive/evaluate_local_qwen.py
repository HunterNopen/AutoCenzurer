import sys
sys.path.append('./')

import logging
logging.basicConfig(
    filename="inference.log",
    level=logging.INFO,      
    encoding="utf-8",
)

logger = logging.getLogger(__name__)

import csv

from evaluation.pipeline import get_batched_dataset, get_inference_results
from evaluation.local import QwenProcessor, BINARY_LABEL_CONFIG

MULTICLASS_CHECKPOINT='artifacts/qwen-toxic-classifier/checkpoint-400'
BINARY_CHECKPOINT='artifacts/qwen-toxic-classifier/checkpoint-417-binary-2'
TOKENIZER_CHECKPOINT='artifacts/qwen-toxic-classifier/checkpoint-417-binary-2'

def main():
    logger.info("getting data")
    dataset = get_batched_dataset('artifacts/binary_classification.csv', batch_size=16)
    logger.info(f"overall batches number: {len(dataset)}")
    
    processor = QwenProcessor(BINARY_CHECKPOINT, TOKENIZER_CHECKPOINT, BINARY_LABEL_CONFIG)

    logger.info("processing data")
    results = get_inference_results(data_iterator=iter(dataset), process_batch_fn=processor.process_batch)
    
    logging.info("saving results...")
    with open('binary_classification_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['y_true', 'y_pred'])
        writer.writerows(zip(results.y_true, results.y_pred))

if __name__ == "__main__":
    main()