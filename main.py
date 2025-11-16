import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'pipelines'))

from pipelines.data_pipeline import run_data_pipeline
from pipelines.training_pipeline import run_training_pipeline
from pipelines.inference_pipeline import run_inference_pipeline
from src.utils import load_config


def main():
    """
    Main entry point for running the Dementia Prediction project pipelines.
    """
    parser = argparse.ArgumentParser(description="Dementia Prediction ML Pipeline Execution.")
    parser.add_argument('pipeline', choices=['data', 'train', 'inference', 'all'],
                        help="The pipeline to run: 'data', 'train', 'inference', or 'all'.")
    args = parser.parse_args()

    config = load_config()

    if args.pipeline == 'data' or args.pipeline == 'all':
        print("\n--- Starting Data Pipeline ---")
        run_data_pipeline()

    if args.pipeline == 'train' or args.pipeline == 'all':
        print("\n--- Starting Training Pipeline ---")
        if not os.path.exists(config['paths']['X_train']):
            print("Warning: Data artifacts not found. Running data pipeline first.")
            run_data_pipeline()
        run_training_pipeline()

    if args.pipeline == 'inference' or args.pipeline == 'all':
        print("\n--- Starting Inference Demonstration ---")
        if not os.path.exists(config['paths']['model_path']):
            print("Warning: Trained model not found. Running training pipeline first.")
            if not os.path.exists(config['paths']['X_train']):
                run_data_pipeline()
            run_training_pipeline()
        run_inference_pipeline()


if __name__ == "__main__":
    main()
