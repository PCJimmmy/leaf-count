.PHONY: requirements data sync_data_from_s3 sync_data_to_s3 clean
.DEFAULT_GOAL := all


#################################################################################
# GLOBALS                                                                       #
#################################################################################

SHELL := /bin/bash
BUCKET = opo-data-science
PROFILE = default
PROJECT_NAME = leaf-count
PYTHON_INTERPRETER = python3
ENVS_DIR := ~/envs
VIRTUALENV := deep_learning_3.7.0
VIRTUALENV_ACTIVATE := $(ENVS_DIR)/$(VIRTUALENV)/bin/activate


#################################################################################
# COMMANDS                                                                      #
#################################################################################

all: requirements sync_data_from_s3

$(ENVS_DIR):
	mkdir $(ENVS_DIR); \
	$(PYTHON_INTERPRETER) -m venv $(ENVS_DIR)/$(VIRTUALENV); \

## Install dependencies
requirements: | $(ENVS_DIR) 
	source $(VIRTUALENV_ACTIVATE); \
	$(PYTHON_INTERPRETER) -m pip install pip --upgrade --force-reinstall; \
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel; \
	$(PYTHON_INTERPRETER) -m pip install --force-reinstall --no-cache-dir -r requirements.txt; \
	wget http://us.download.nvidia.com/tesla/375.66/nvidia-diag-driver-local-repo-ubuntu1604_375.66-1_amd64.deb; \
	sudo dpkg -i nvidia-diag-driver-local-repo-ubuntu1604_375.66-1_amd64.deb; \
	sudo apt-get update; \
	sudo apt-get -y install cuda-drivers; \
	rm nvidia-diag-driver-local-repo-ubuntu1604_375.66-1_amd64.deb; \

## Make dataset
data: | sync_data_from_s3
	source $(VIRTUALENV_ACTIVATE); \
	$(PYTHON_INTERPRETER) src/data/make_dataset.py; \

## Download raw data from S3
sync_data_from_s3:
	source $(VIRTUALENV_ACTIVATE); \
	if [[ default = $(PROFILE) ]]; \
		then AWS_CONFIG_FILE=~/.aws_config aws s3 sync s3://$(BUCKET)/$(PROJECT_NAME)/data/raw/ data/raw/; \
	else AWS_CONFIG_FILE=~/.aws_config aws s3 sync s3://$(BUCKET)/$(PROJECT_NAME)/data/raw/ data/raw/ --profile $(PROFILE); \
	fi; \

## Upload processed data to S3
sync_data_to_s3:
	source $(VIRTUALENV_ACTIVATE); \
	if [[ default = $(PROFILE) ]]; \
		then AWS_CONFIG_FILE=~/.aws_config aws s3 sync data/processed/ s3://$(BUCKET)/$(PROJECT_NAME)/data/processed/; \
	else AWS_CONFIG_FILE=~/.aws_config aws s3 sync data/processed/ s3://$(BUCKET)/$(PROJECT_NAME)/data/processed/ --profile $(PROFILE); \
	fi; \

## Test python environment is setup correctly
test_environment:
	source $(VIRTUALENV_ACTIVATE); \
	$(PYTHON_INTERPRETER) test_environment.py; \

## Delete all compiled Python files
clean:
	source $(VIRTUALENV_ACTIVATE); \
	find . -type f -name "*.py[co]" -delete; \
	find . -type d -name "__pycache__" -delete; \


