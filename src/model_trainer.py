import os

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier

from src.utils import load_config


def build_stacked_pipeline(config):
    """
    Constructs the full preprocessing and stacked ensemble pipeline
    without fitting, using parameters from the config.
    """
    feat_config = config['features']
    model_config = config['model_params']
    random_state = model_config['random_state']

    categorical_features = feat_config['categorical_features']
    continuous_features = feat_config['continuous_features']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), continuous_features)
    ], remainder='passthrough')

    xgb_params = {**model_config['xgb_params'], 'random_state': random_state}
    lgb_params = {**model_config['lgb_params'], 'random_state': random_state}
    cat_params = {**model_config['cat_params'], 'random_state': random_state}
    rf_params = {**model_config['rf_params'], 'random_state': random_state}

    base_models = [
        ('xgb', XGBClassifier(**xgb_params)),
        ('lgb', LGBMClassifier(**lgb_params)),
        ('cat', CatBoostClassifier(**cat_params)),
        ('rf', RandomForestClassifier(**rf_params, n_jobs=-1))
    ]

    stack_classifier = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(**model_config['final_estimator_params']),
        cv=5,
        passthrough=True,
        n_jobs=-1
    )

    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=model_config['smote_random_state'])),
        ('stack', stack_classifier)
    ])

    return pipeline


def train_stacked_model(config):
    """
    Trains the stacked ensemble model and saves the fitted pipeline.
    """
    paths_config = config['paths']

    X_train = pd.read_csv(paths_config['X_train'])
    y_train = pd.read_csv(paths_config['y_train']).squeeze()

    pipeline = build_stacked_pipeline(config)

    print("Starting model training...")
    pipeline.fit(X_train, y_train)
    print("Model training complete.")

    os.makedirs(os.path.dirname(paths_config['model_path']), exist_ok=True)
    joblib.dump(pipeline, paths_config['model_path'])
    print(f"Trained Stacked Ensemble Model saved to: {paths_config['model_path']}")


if __name__ == '__main__':
    config = load_config()
    train_stacked_model(config)
