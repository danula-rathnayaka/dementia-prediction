.PHONY: all clean install train-pipeline data-pipeline run-all help

PYTHON = python
VENV = .venv\Scripts\activate
VENV_PYTHON = .venv\Scripts\python.exe

all: run-all

help:
	@echo "Available targets:"
	@echo "  make install             - Install project dependencies and set up environment."
	@echo "  make data-pipeline       - Run the data preparation pipeline (Ingest, Clean, Split)."
	@echo "  make train-pipeline      - Run the model training and evaluation pipeline."
	@echo "  make inference-pipeline  - Run the model inference demonstration."
	@echo "  make run-all             - Run all pipelines in sequence (data, train, inference)."
	@echo "  make clean               - Clean up artifacts and models."

install:
	@echo Installing project dependencies and setting up environment...
	@echo Creating virtual environment...
	@$(PYTHON) -m venv .venv
	@echo Activating virtual environment and installing dependencies...
	@$(VENV_PYTHON) -m pip install --upgrade pip
	@$(VENV_PYTHON) -m pip install -r requirements.txt
	@echo Installation completed successfully!
	@echo To activate the virtual environment manually, run: call .venv\Scripts\activate

data-pipeline:
	@echo Start running data pipeline (Ingestion, Cleaning, Splitting)...
	@call $(VENV) && $(PYTHON) pipelines\data_pipeline.py
	@echo Data pipeline completed successfully!

train-pipeline:
	@echo Running training and evaluation pipeline...
	@call $(VENV) && $(PYTHON) pipelines\training_pipeline.py
	@echo Training pipeline completed successfully!

inference-pipeline:
	@echo Running inference pipeline demonstration...
	@call $(VENV) && $(PYTHON) pipelines\inference_pipeline.py
	@echo Inference pipeline completed successfully!

run-all: data-pipeline train-pipeline inference-pipeline
	@echo ========================================
	@echo All pipelines completed successfully!
	@echo ========================================

clean:
	@echo Cleaning up artifacts, models, and virtual environment...
	@if exist .venv rmdir /s /q .venv
	@if exist artifacts\data rmdir /s /q artifacts\data
	@if exist artifacts\preprocessor rmdir /s /q artifacts\preprocessor
	@if exist models rmdir /s /q models
	@echo Cleanup completed!