# Dementia Risk Prediction using Non-Medical Features

## 🧠 Project Overview

This project implements a complete Machine Learning Operations (MLOps) pipeline designed to predict an individual's risk of dementia using **exclusively non-medical, demographic, and lifestyle variables**. This work was developed in response to a hackathon challenge focused on exploring the predictive power of non-clinical data for a major global health issue.

The final system delivers a robust binary classification model, providing a probability score (0-100%) for dementia risk based on features like education, social context, and basic lifestyle factors from the NACC cohort data.

## 📊 Dataset

The model is trained on a curated subset of the **National Alzheimer's Coordinating Center (NACC) cohort**. The dataset provides detailed participant visit information, including a wide array of non-medical features and a binary target label indicating the presence or absence of dementia.

## 🧪 Analysis & Experimentation

All exploratory data analysis (EDA), feature engineering trials, model development, and comparative evaluations are fully documented in the project notebook:

➡️ [**notebooks/Dementia_Prediction_Trails.ipynb**](https://github.com/danula-rathnayaka/dementia-prediction/blob/main/notebooks/Dementia_Prediction_Trials.ipynb)

The automated pipelines below leverage the best-performing ensemble model architecture identified during the experimentation phase.

## 🚀 MLOps Pipeline Execution

The entire machine learning workflow from data ingestion to model inference is managed via a simple `Makefile`. The recommended execution environment is a terminal (PowerShell or Command Prompt) with `make` and Python installed.

### Prerequisites

1. Python 3.8+

2. `make` utility (or equivalent build tool)

3. The project's `requirements.txt` file.

### Setup and Running

Use the following commands to set up the environment and run the full pipeline:

| **Command** | **Description** |
| :--- | :--- |
| `make install` | **[MANDATORY FIRST STEP]** Creates the virtual environment (`.venv`) and installs all dependencies from `requirements.txt`. |
| `make run-all` | Executes the entire MLOps workflow sequentially (Data $\rightarrow$ Train $\rightarrow$ Inference). |
| `make clean` | Removes the virtual environment and all generated artifacts (`models/`, `artifacts/data/`, etc.). |

### Individual Pipeline Targets

| **Command** | **Description** |
| :--- | :--- |
| `make data-pipeline` | Runs `pipelines/data_pipeline.py`. Handles ingestion, cleaning, feature engineering, and train/test splitting. Outputs split data to `artifacts/data/`. |
| `make train-pipeline` | Runs `pipelines/training_pipeline.py`. Loads split data, applies preprocessing, trains the best-performing model, and saves the final model artifact to `models/`. |
| `make inference-pipeline` | Runs `pipelines/inference_pipeline.py`. Loads the trained model and demonstrates predictions on the test set. |

### Manual Virtual Environment Activation

To work on the code interactively, activate the environment manually:

```bash
.venv\Scripts\activate