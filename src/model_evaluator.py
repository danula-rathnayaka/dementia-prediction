import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from src.utils import load_config


def evaluate_model(config):
    """
    Loads the trained model and test data, then evaluates the model
    using the specified classification threshold.
    """
    paths_config = config['paths']
    threshold = config['model_params']['classification_threshold']

    pipeline = joblib.load(paths_config['model_path'])
    X_test = pd.read_csv(paths_config['X_test'])
    y_test = pd.read_csv(paths_config['y_test']).squeeze()

    y_prob = pipeline.predict_proba(X_test)[:, 1]

    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_prob)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("\n\n" + "=" * 50)
    print("         MODEL EVALUATION (Stacked Ensemble)      ")
    print("=" * 50)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"\nClassification Report (Threshold={threshold}):\n{class_report}")
    print(f"\nConfusion Matrix (Threshold={threshold}):\n{conf_matrix}")
    print("=" * 50)


if __name__ == '__main__':
    config = load_config()
    evaluate_model(config)
