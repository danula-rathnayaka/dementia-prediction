import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_evaluator import evaluate_model
from src.model_trainer import train_stacked_model
from src.utils import load_config


def run_training_pipeline():
    """
    Executes the full training pipeline: model training and evaluation.
    """
    print("--- Running Training Pipeline ---")
    config = load_config()

    train_stacked_model(config)

    evaluate_model(config)

    print("--- Training Pipeline Complete ---")


if __name__ == '__main__':
    run_training_pipeline()
