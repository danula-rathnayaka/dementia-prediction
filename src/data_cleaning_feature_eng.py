import os

import numpy as np
import pandas as pd

from src.utils import load_config


def clean_and_engineer_features(config, raw_data_path=None):
    """
    Performs data cleaning, handles special codes, imputes missing values,
    and creates new features based on the notebook steps.
    """
    paths_config = config['paths']
    feat_config = config['features']

    input_path = raw_data_path if raw_data_path else paths_config['raw_data']
    df = pd.read_csv(input_path)

    df = df[feat_config['selected_columns']].copy()

    df_base = df.sort_values(['NACCID', 'VISITYR']).drop_duplicates('NACCID', keep='first')

    nan_map = {
        'EDUC': [99], 'MARISTAT': [9], 'NACCLIVS': [9], 'INDEPEND': [9],
        'NACCFAM': [9], 'NACCMOM': [9], 'NACCDAD': [9], 'TOBAC30': [9],
        'TOBAC100': [9], 'ALCOCCAS': [9], 'ALCFREQ': [8, 9], 'CBSTROKE': [9],
        'HATTYEAR': [9999], 'CVHATT': [9], 'DIABETES': [9], 'HYPERTEN': [9],
        'DEP2YRS': [9], 'ANXIETY': [9], 'HEARING': [9]
    }
    zero_map = {
        'NACCFAM': [-4], 'NACCMOM': [-4], 'NACCDAD': [-4], 'TOBAC30': [-4],
        'TOBAC100': [-4], 'ALCOCCAS': [-4], 'ALCFREQ': [-4], 'CBSTROKE': [-4],
        'HATTYEAR': [8888], 'CVHATT': [-4], 'DIABETES': [-4], 'HYPERTEN': [-4],
        'DEP2YRS': [-4], 'ANXIETY': [-4], 'HEARING': [-4]
    }

    for col, vals in nan_map.items():
        if col in df_base.columns: df_base[col] = df_base[col].replace(vals, np.nan)
    for col, vals in zero_map.items():
        if col in df_base.columns: df_base[col] = df_base[col].replace(vals, 0)

    df_base['AGE'] = (df_base['VISITYR'] - df_base['BIRTHYR']).clip(lower=50, upper=100)
    df_base['SEX'] = df_base['SEX'].map({1: 1, 2: 0})

    int_cols = ['TOBAC30', 'TOBAC100', 'HEARING', 'ALCOCCAS', 'ALCFREQ']
    for col in int_cols:
        if col in df_base.columns:
            df_base[col] = df_base[col].fillna(0).astype(int)

    health_cols = ['CVHATT', 'CBSTROKE', 'DIABETES', 'HYPERTEN', 'DEP2YRS', 'ANXIETY']
    for col in health_cols:
        if col in df_base.columns:
            df_base[col] = df_base[col].map({1: 1, 2: 0}).fillna(0).astype(int)

    df_base[['NACCFAM', 'NACCMOM', 'NACCDAD']] = df_base[['NACCFAM', 'NACCMOM', 'NACCDAD']].fillna(0)

    df_proc = df_base.copy()
    df_proc['EDUC'] = df_proc['EDUC'].fillna(df_proc['EDUC'].median())
    df_proc['MARISTAT'] = df_proc['MARISTAT'].fillna(df_proc['MARISTAT'].mode()[0])
    df_proc['INDEPEND'] = df_proc['INDEPEND'].fillna(df_proc['INDEPEND'].mode()[0])
    df_proc['NACCLIVS'] = df_proc['NACCLIVS'].fillna(df_proc['NACCLIVS'].mode()[0])
    df_proc['HATTYEAR'] = df_proc['HATTYEAR'].replace(-4, 0).fillna(0)

    final_cols = feat_config['binary_features'] + feat_config['categorical_features'] + feat_config[
        'continuous_features'] + [feat_config['target']]

    df_final = df_proc[[col for col in final_cols if col in df_proc.columns]]

    output_path = paths_config['cleaned_data']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"Cleaned and engineered data saved to: {output_path}")
    return output_path


if __name__ == '__main__':
    config = load_config()
    clean_and_engineer_features(config)
