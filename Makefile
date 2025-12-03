
# ======================================================================
# Makefile for MLOps Training Project 
# ======================================================================

# ---------------------------- VARIABLES --------------------------------
INPUT_DATA_PATH   ?= datastores/raw_csv_data/housing.csv
OUTPUT_FILENAME   ?= clean_housing.csv
INPUT_TRAIN_DATA  ?= datastores/splits_data/train_data.csv
INPUT_TEST_DATA   ?= datastores/splits_data/test_data.csv
MODEL_FILENEME    ?= modelstores/LinearRegression.joblib
CONDA_ENV         ?= ml_env


# --------------------------- DEFAULT TARGETS ------------------------------
.PHONY: env_update install_dependencies update_dependencies clean split train pipeline

# ======================================================================
# ENVIRONMENT MANAGEMENT
# ======================================================================

# Update conda environment
env_update:
	@echo "=> Updating conda environment from conda.yaml (env: $(CONDA_ENV))"
	conda env update -f conda.yaml 
	@echo "=> Conda environment '$(CONDA_ENV)' updated successfully."

# ======================================================================
# DEPENDENCY MANAGEMENT
# ======================================================================

# Install project dependencies (Poetry)
install_dependencies:
	@echo "=> Installing project dependencies..."
	poetry install
	@echo "=> Dependencies installed successfully."

# Update all dependencies (Poetry)
update_dependencies:
	@echo "=> Updating project dependencies..."
	poetry update
	@echo "=> Dependencies updated successfully."

# ======================================================================
# DATA PREPROCESSING, SPLITING & TRAINING
# ======================================================================

# Run data preprocessing script
clean:
	@echo "=> Running data preprocessing..."
	## your code here
	@echo "=> Data preprocessing completed. Clean data saved to $(OUTPUT_FILENAME)."

# Run data preprocessing script
split:
	@echo "=> Running splits data ..."
	## your code here
	@echo "=> Splits data completed. Clean data saved to $(OUTPUT_FILENAME)."

# Run training script
train:
	@echo "=> Running train model..."
	## your code here
	@echo "=> train model completed successfully."	



# ======================================================================
# ALL-IN-ONE WORKFLOW : local ci pipeline
# ======================================================================

pipeline: ## your code here
	@echo "All tasks completed successfully."

