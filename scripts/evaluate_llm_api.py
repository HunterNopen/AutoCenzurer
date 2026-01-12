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

from evaluation.groq import GroqBinaryProcessor
from evaluation.pipeline import get_batched_dataset, get_inference_results
from static.config import LLM_PROMPT_BINARY_CLASSIFICATION

def main():
    logger.info("getting data")
    dataset = get_batched_dataset('binary_classification.csv', batch_size=5)
    dataset = dataset.take(1)
    
    llm_binary_processor = GroqBinaryProcessor(system_prompt=LLM_PROMPT_BINARY_CLASSIFICATION, max_concurrent=2)
    logger.info("processing data")
    results = get_inference_results(data_iterator=iter(dataset), process_batch_fn=llm_binary_processor.process_batch)
    
    logging.info("saving results...")
    with open('classification_inference_result.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['y_true', 'y_pred'])
        writer.writerows(zip(results.y_true, results.y_pred))
    
    llm_binary_processor.export_metadata('classification_inference_metadata_result.csv')
    
if __name__ == "__main__":
    main()