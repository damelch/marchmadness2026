.PHONY: build test simulate optimize-day1 optimize-day2 optimize-all clean lint format

IMAGE := marchmadness
DATA := $(CURDIR)/data
CONFIG := $(CURDIR)/config.yaml
ENTRIES := $(CURDIR)/entries
DOCKER_RUN := docker run --rm \
	-v "$(DATA):/app/data" \
	-v "$(CONFIG):/app/config.yaml" \
	-v "$(ENTRIES):/app/entries"

# ── Build ────────────────────────────────────────────────────────────
build:
	docker build -t $(IMAGE) .

# ── Tests ────────────────────────────────────────────────────────────
test: build
	docker run --rm --entrypoint pytest $(IMAGE) -v

# ── Simulation ───────────────────────────────────────────────────────
simulate: build
	$(DOCKER_RUN) $(IMAGE) simulate

# ── Optimizer ────────────────────────────────────────────────────────
# Defaults — override on command line: make optimize-day1 POOL_SIZE=5000
METHOD     ?= hybrid
POOL_SIZE  ?=
NUM_ENTRIES ?=
MAX_ENTRIES ?=

# Build the optional CLI flags
OPT_FLAGS := --method $(METHOD)
ifneq ($(POOL_SIZE),)
  OPT_FLAGS += --pool-size $(POOL_SIZE)
endif
ifneq ($(NUM_ENTRIES),)
  OPT_FLAGS += --num-entries $(NUM_ENTRIES)
endif
ifneq ($(MAX_ENTRIES),)
  OPT_FLAGS += --max-entries $(MAX_ENTRIES)
endif

optimize-day1: build
	$(DOCKER_RUN) $(IMAGE) optimize --day 1 $(OPT_FLAGS)

optimize-day2: build
	$(DOCKER_RUN) $(IMAGE) optimize --day 2 $(OPT_FLAGS)

optimize-all: build
	$(DOCKER_RUN) $(IMAGE) optimize --day 1 $(OPT_FLAGS)
	$(DOCKER_RUN) $(IMAGE) optimize --day 2 $(OPT_FLAGS)

optimize-day%: build
	$(DOCKER_RUN) $(IMAGE) optimize --day $* $(OPT_FLAGS)

# ── Code Quality ────────────────────────────────────────────────────
lint: build
	docker run --rm --entrypoint ruff $(IMAGE) check .

format: build
	docker run --rm --entrypoint ruff $(IMAGE) format --check .

# ── Utilities ────────────────────────────────────────────────────────
reset:
	rm -f entries/state.json

clean: reset
	docker rmi $(IMAGE) 2>/dev/null || true
