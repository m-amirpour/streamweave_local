SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
.NOTPARALLEL:

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
-include $(ROOT)/streamweave.mk

PYTHON ?= python3
RUNNER ?= $(ROOT)/streamweave_revised.py
PREFLIGHT := $(ROOT)/scripts/preflight.py
REPORTER := $(ROOT)/scripts/summarize_results.py

DATA_DIR ?= $(ROOT)/data
RESULTS_DIR ?= $(ROOT)/results/streamweave_paper
SEARCH_MODE ?= full_factorial
DATASETS ?= Covertype,Skin,N-BaIoT,NSL-KDD,CreditCardFraud,Adult,BankMarketing,Electricity
REQUIRE_CUDA ?= 1

CUDA_FLAG := $(if $(filter 1 true yes TRUE YES,$(REQUIRE_CUDA)),--require-cuda,)

# Pass all configured settings into the existing runner consistently.
export STREAMWEAVE_DATA_DIR := $(abspath $(DATA_DIR))
export STREAMWEAVE_RESULTS_DIR := $(abspath $(RESULTS_DIR))
export STREAMWEAVE_SEARCH_MODE := $(SEARCH_MODE)
export STREAMWEAVE_DATASETS := $(DATASETS)
export STREAMWEAVE_SCREEN_TRAIN_SIZE := $(SCREEN_TRAIN_SIZE)
export STREAMWEAVE_SCREEN_EPOCHS := $(SCREEN_EPOCHS)
export STREAMWEAVE_SCREEN_KD_EPOCHS := $(SCREEN_KD_EPOCHS)
export STREAMWEAVE_SCREEN_EVAL_MAX := $(SCREEN_EVAL_MAX)
export STREAMWEAVE_SCREEN_NEURAL_BUDGET := $(SCREEN_NEURAL_BUDGET)
export STREAMWEAVE_SCREEN_MAX_FAILURE_FRACTION := $(SCREEN_MAX_FAILURE_FRACTION)
export STREAMWEAVE_TOP_K := $(TOP_K)
export STREAMWEAVE_CONFIRM_TRAIN_SIZE := $(CONFIRM_TRAIN_SIZE)
export STREAMWEAVE_CONFIRM_EPOCHS := $(CONFIRM_EPOCHS)
export STREAMWEAVE_CONFIRM_KD_EPOCHS := $(CONFIRM_KD_EPOCHS)
export STREAMWEAVE_SEEDS := $(SEEDS)
export STREAMWEAVE_NBAIOT_ROWS := $(NBAIOT_ROWS)
export STREAMWEAVE_NUM_WORKERS := $(NUM_WORKERS)
export STREAMWEAVE_MASTER_SEED := $(MASTER_SEED)
export STREAMWEAVE_REFRESH_EPOCHS := $(REFRESH_EPOCHS)
export STREAMWEAVE_ONLINE_EPOCHS := $(ONLINE_EPOCHS)
export STREAMWEAVE_STREAM_BATCH_SIZE := $(STREAM_BATCH_SIZE)
export STREAMWEAVE_BATCH_INTERVAL := $(BATCH_INTERVAL)
export STREAMWEAVE_MEMORY_MAX := $(MEMORY_MAX)

.PHONY: help check smoke plan run pilot all report status

help:
	@printf '\nStreamWeave automation\n\n'
	@printf '  make check       Check Python packages, CUDA, paths, and writable storage\n'
	@printf '  make smoke       Run the runner’s fast built-in smoke tests\n'
	@printf '  make plan        Print/write the resolved run plan; does not train models\n'
	@printf '  make run         Run screening -> confirmation -> final evaluation\n'
	@printf '  make pilot       Run one-dataset pairwise end-to-end pilot (Adult)\n'
	@printf '  make all         Check, smoke-test, preview plan, run, and summarize\n'
	@printf '  make status      Show the latest saved run state\n'
	@printf '  make report      Rebuild the results overview from saved output files\n\n'
	@printf 'Common overrides: make all DATASETS=Adult REQUIRE_CUDA=0 SEARCH_MODE=pairwise\n'
	@printf 'Settings file: %s/streamweave.mk\n' '$(ROOT)'
	@printf 'Default search: %s | datasets: %s\n\n' '$(SEARCH_MODE)' '$(DATASETS)'

check:
	@"$(PYTHON)" "$(PREFLIGHT)" \
		--runner "$(RUNNER)" \
		--data-dir "$(STREAMWEAVE_DATA_DIR)" \
		--results-dir "$(STREAMWEAVE_RESULTS_DIR)" \
		--search-mode "$(SEARCH_MODE)" \
		--datasets "$(DATASETS)" \
		$(CUDA_FLAG)

smoke: check
	@echo '== Running built-in smoke tests =='
	@"$(PYTHON)" "$(RUNNER)" --smoke-test $(CUDA_FLAG)

plan: check
	@echo '== Writing/printing experiment plan (no model training) =='
	@"$(PYTHON)" "$(RUNNER)" --dry-run $(CUDA_FLAG)

# The current Python runner performs all three scientific stages in one call.
# We therefore do not fake separate 'screen' or 'confirm' commands that the
# runner does not support. The wrapper also validates FINAL_MANIFEST.json.
run: check
	@echo '== Running complete StreamWeave experiment =='
	@set +e; \
	start_epoch="$$(date +%s)"; \
	"$(PYTHON)" "$(RUNNER)" $(CUDA_FLAG); run_rc=$$?; \
	"$(PYTHON)" "$(REPORTER)" \
		--results-dir "$(STREAMWEAVE_RESULTS_DIR)" \
		--expected-datasets "$(DATASETS)" \
		--since-epoch "$$start_epoch" --strict; report_rc=$$?; \
	if [[ $$run_rc -ne 0 ]]; then exit $$run_rc; fi; \
	exit $$report_rc

pilot:
	@$(MAKE) all DATASETS=Adult SEARCH_MODE=pairwise

all: check smoke plan run

report:
	@"$(PYTHON)" "$(REPORTER)" --results-dir "$(STREAMWEAVE_RESULTS_DIR)"

status:
	@"$(PYTHON)" "$(REPORTER)" --results-dir "$(STREAMWEAVE_RESULTS_DIR)"
