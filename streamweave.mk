# StreamWeave experiment settings.
# Change values here instead of maintaining a long list of shell exports.

# Exhaustive default: 5 BPU x 6 SPU x 7 transfer methods x 7 CU methods
# = 1,470 screening rows per dataset. Set SEARCH_MODE=pairwise for a smaller pilot.
SEARCH_MODE ?= full_factorial
DATASETS ?= Covertype,Skin,N-BaIoT,NSL-KDD,CreditCardFraud,Adult,BankMarketing,Electricity

# Fail early if PyTorch cannot see CUDA. Set REQUIRE_CUDA=0 to allow CPU fallback.
REQUIRE_CUDA ?= 1

# Paths (absolute paths are built in the Makefile).
DATA_DIR ?= $(ROOT)/data
RESULTS_DIR ?= $(ROOT)/results/streamweave_paper

# Tiny screening and confirmation budgets; these are starting budgets, not a
# guarantee of convergence or paper-ready evidence.
SCREEN_TRAIN_SIZE ?= 2000
SCREEN_EPOCHS ?= 1
SCREEN_KD_EPOCHS ?= 1
SCREEN_EVAL_MAX ?= 2500
SCREEN_NEURAL_BUDGET ?= 43
SCREEN_MAX_FAILURE_FRACTION ?= 0.01
TOP_K ?= 3

CONFIRM_TRAIN_SIZE ?= 8000
CONFIRM_EPOCHS ?= 5
CONFIRM_KD_EPOCHS ?= 1
SEEDS ?= 0,1,2

# Resource controls.
NBAIOT_ROWS ?= 855932
NUM_WORKERS ?= 0
MASTER_SEED ?= 20261010
REFRESH_EPOCHS ?= 2
ONLINE_EPOCHS ?= 1
STREAM_BATCH_SIZE ?= 256
BATCH_INTERVAL ?= 8192
MEMORY_MAX ?= 4096
