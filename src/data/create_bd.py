import os
import sys
import pandas as pd
sys.path.append('..')
from tqdm import tqdm
from unstructured_ingest.v2.pipeline.pipeline import Pipeline
from unstructured_ingest.v2.interfaces import ProcessorConfig
from unstructured_ingest.v2.processes.connectors.local import (
    LocalIndexerConfig,
    LocalDownloaderConfig,
    LocalConnectionConfig,
    LocalUploaderConfig
)
from unstructured_ingest.v2.processes.partitioner import PartitionerConfig
from src.utils.paths import get_project_path

def local_parse_data(directory_with_pdfs: str, directory_with_results: str):
    
    """
    Parse PDF files in a directory and its subdirectories, and save the results 
    in another directory.

    Args:
        directory_with_pdfs (str): Path to a directory containing PDF files.
        directory_with_results (str): Path to a directory where the parsed results
            will be saved.

    Returns:
        None
    """
    pdf_files = []
    for root, dirs, files in os.walk(directory_with_pdfs):
        for file in files:
            if file.endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))


    for pdf_file in tqdm(pdf_files):
        Pipeline.from_configs(
            context=ProcessorConfig(num_processes=3),
            indexer_config=LocalIndexerConfig(input_path=pdf_file),
            downloader_config=LocalDownloaderConfig(),
            source_connection_config=LocalConnectionConfig(),
            partitioner_config=PartitionerConfig(
                strategy="hi_res",
                additional_partition_args={
                    "preserve_formatting": True,
                    "split_pdf_page": True,
                    "split_pdf_concurrency_level": 15,
                },
            ),
            uploader_config=LocalUploaderConfig(output_dir=directory_with_results)
        ).run()


if __name__ == "__main__":
    directory_with_pdfs = os.path.join(get_project_path(), 'data', 'Articles for the database')
    directory_with_results = os.path.join(get_project_path(), 'data', 'parsed_pages')

    local_parse_data(directory_with_pdfs, directory_with_results)