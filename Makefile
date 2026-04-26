IMAGE ?= quay.io/jupyter/pyspark-notebook
WORKSPACE_ROOT := $(abspath ..)
CONTAINER_ROOT := /home/jovyan/work
PROJECT_ROOT := $(CONTAINER_ROOT)/ac2
AIRBNB_TOTAL_CSV := $(PROJECT_ROOT)/data/raw/airbnb_rio/total_data.csv
AIRBNB_PARQUET_PATH := $(PROJECT_ROOT)/data/processed/airbnb_rio/listings_clean.parquet
DATASET_PATH := $(AIRBNB_PARQUET_PATH)
PORTS ?= -p 8888:8888 -p 4040:4040 -p 7077:7077
TRAIN_RATIO ?= 0.8
SEED ?= 42
SPARK_MASTER ?=
PYTHON ?= python
PARQUET_PARTITIONS ?= 32
MODEL_SAMPLE_ROWS ?= 5000

DOCKER_BASE = docker run --rm \
	-v "$(WORKSPACE_ROOT):$(CONTAINER_ROOT)" \
	-w "$(PROJECT_ROOT)"

DOCKER_ENV = \
		-e AC2_PARQUET_PATH=$(DATASET_PATH) \
		-e AC2_TRAIN_RATIO=$(TRAIN_RATIO) \
		-e AC2_SEED=$(SEED) \
		-e AC2_MODEL_SAMPLE_ROWS=$(MODEL_SAMPLE_ROWS) \
		$(if $(SPARK_MASTER),-e AC2_SPARK_MASTER=$(SPARK_MASTER),)

.PHONY: help doctor notebook shell parquet-airbnb run eda models models-sample

help: ## Show available targets and useful variables
	@printf "AC2 Make targets\n\n"
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(firstword $(MAKEFILE_LIST)) | sort | awk 'BEGIN {FS = ":.*## "}; {printf "  %-14s %s\n", $$1, $$2}'
	@printf "\nVariables you can override:\n"
	@printf "  IMAGE=%s\n" "$(IMAGE)"
	@printf "  TRAIN_RATIO=%s\n" "$(TRAIN_RATIO)"
	@printf "  SEED=%s\n" "$(SEED)"
	@printf "  SPARK_MASTER=%s\n" "$(SPARK_MASTER)"
	@printf "  PARQUET_PARTITIONS=%s\n" "$(PARQUET_PARTITIONS)"
	@printf "  MODEL_SAMPLE_ROWS=%s\n" "$(MODEL_SAMPLE_ROWS)"

doctor: ## Check whether the selected image exposes pyspark in its Python environment
	$(DOCKER_BASE) \
		$(IMAGE) \
		bash -lc 'which python || true; which python3 || true; python -c "import os,sys,pyspark; print(\"python_executable=\", sys.executable); print(\"python_version=\", sys.version.replace(\"\\n\", \" \")); print(\"SPARK_HOME=\", os.environ.get(\"SPARK_HOME\")); print(\"pyspark=\", pyspark.__file__)"'

notebook: ## Start an interactive Spark/Jupyter container in the professor-compatible image
	docker run --rm -it \
		$(PORTS) \
		-v "$(WORKSPACE_ROOT):$(CONTAINER_ROOT)" \
		-w "$(PROJECT_ROOT)" \
		$(IMAGE)

shell: ## Open a bash shell inside the professor-compatible image at ac2/
	docker run --rm -it \
		-v "$(WORKSPACE_ROOT):$(CONTAINER_ROOT)" \
		-w "$(PROJECT_ROOT)" \
		$(IMAGE) \
		bash

parquet-airbnb: ## Convert the validated Airbnb Rio total_data.csv to cleaned Parquet
	$(DOCKER_BASE) \
		$(IMAGE) \
		bash -lc '$(PYTHON) scripts/convert_airbnb_csv_to_parquet.py --input "$(AIRBNB_TOTAL_CSV)" --output "$(AIRBNB_PARQUET_PATH)" --partitions "$(PARQUET_PARTITIONS)"'

eda: ## Run the Spark EDA script in Docker
	$(MAKE) run

models: ## Run the Spark price prediction notebook script in Docker
	$(MAKE) run

models-sample: ## Run the notebook script with a smaller random sample
	$(MAKE) run MODEL_SAMPLE_ROWS=1000

run: ## Run the complete Airbnb price prediction notebook script in Docker
	$(DOCKER_BASE) \
		$(DOCKER_ENV) \
		$(IMAGE) \
		bash -lc '$(PYTHON) notebooks/01_airbnb_price_prediction.py'
