import joblib
import pandas as pd

from src.utils import load_config


def make_inference(input_data_df: pd.DataFrame, config, threshold: float = None):
    """
    Loads the trained model and performs inference on new input data.

    Args:
        input_data_df: DataFrame containing the raw features for prediction.
        config: Loaded project configuration.
        threshold: The classification probability threshold. Uses config value if None.

    Returns:
        A list of predicted classes (0 or 1).
    """
    paths_config = config['paths']

    if threshold is None:
        threshold = config['model_params']['classification_threshold']

    pipeline = joblib.load(paths_config['model_path'])

    y_prob = pipeline.predict_proba(input_data_df)[:, 1]

    y_pred = (y_prob >= threshold).astype(int)

    return y_pred.tolist(), y_prob.tolist()


if __name__ == '__main__':
    config = load_config()
    paths_config = config['paths']

    try:
        X_test = pd.read_csv(paths_config['X_test'])
        sample_input = X_test.head(5).copy()

        predictions, probabilities = make_inference(sample_input, config)

        print("\n--- Inference Sample ---")
        print(sample_input)
        print(f"\nPredicted Classes: {predictions}")
        print(f"Probabilities: {[f'{p:.4f}' for p in probabilities]}")

    except FileNotFoundError:
        print("Please run the full data and training pipelines first to generate X_test and the model.")
