import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

from src.model_inference import make_inference
from src.utils import load_config


def run_inference_pipeline():
    """
    Demonstrates model inference using a sample of the test data.
    """
    print("--- Running Inference Pipeline ---")
    config = load_config()
    paths_config = config['paths']

    try:
        X_test = pd.read_csv(paths_config['X_test'])
        sample_input = X_test.head(5)

        predictions, probabilities = make_inference(sample_input, config)

        print("\n--- Inference Results ---")
        print(f"Input Data Shape: {sample_input.shape}")
        print(f"Predicted Classes (Dementia Risk 1/0): {predictions}")
        print(f"Probabilities: {[f'{p:.4f}' for p in probabilities]}")

    except FileNotFoundError:
        print(
            "\nERROR: Cannot run inference. Please ensure the data and training pipelines have been run successfully.")

    print("--- Inference Pipeline Complete ---")


if __name__ == '__main__':
    run_inference_pipeline()
