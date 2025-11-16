import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_cleaning_feature_eng import clean_and_engineer_features
from src.utils import load_config


def split_data(config, cleaned_data_path=None):
    """
    Splits the cleaned dataset into X_train, X_test, y_train, y_test
    and saves them to the artifacts folder.
    """
    paths_config = config['paths']
    model_config = config['model_params']
    feat_config = config['features']

    # Load cleaned data
    input_path = cleaned_data_path if cleaned_data_path else paths_config['cleaned_data']
    if not os.path.exists(input_path):
        print("Cleaned data not found. Running cleaning and feature engineering...")
        input_path = clean_and_engineer_features(config)

    df = pd.read_csv(input_path)

    all_final_features = feat_config['binary_features'] + feat_config['categorical_features'] + feat_config[
        'continuous_features']

    X = df[[col for col in all_final_features if col in df.columns]]
    y = df[feat_config['target']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=model_config['test_size'],
        stratify=y,
        random_state=model_config['random_state']
    )

    # Save splits
    os.makedirs(os.path.dirname(paths_config['X_train']), exist_ok=True)
    X_train.to_csv(paths_config['X_train'], index=False)
    X_test.to_csv(paths_config['X_test'], index=False)
    y_train.to_csv(paths_config['y_train'], index=False, header=True)
    y_test.to_csv(paths_config['y_test'], index=False, header=True)

    print(f"Data split successful. Train/Test sets saved to {os.path.dirname(paths_config['X_train'])}")


if __name__ == '__main__':
    config = load_config()
    split_data(config)
