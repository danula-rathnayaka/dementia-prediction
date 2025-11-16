import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_cleaning_feature_eng import clean_and_engineer_features
from src.data_ingestion import ingest_data
from src.data_splitter import split_data
from src.utils import load_config


def run_data_pipeline():
    """
    Executes the full data preparation pipeline: ingestion, cleaning,
    feature engineering, and splitting.
    """
    print("--- Running Data Pipeline ---")
    config = load_config()

    raw_path = ingest_data(config)

    cleaned_path = clean_and_engineer_features(config, raw_data_path=raw_path)

    split_data(config, cleaned_data_path=cleaned_path)

    print("--- Data Pipeline Complete ---")


if __name__ == '__main__':
    run_data_pipeline()
