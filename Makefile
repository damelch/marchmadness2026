.PHONY: build test simulate optimize-day1 optimize-day2 optimize-all clean

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
METHOD ?= hybrid

optimize-day1: build
	$(DOCKER_RUN) $(IMAGE) optimize --day 1 --method $(METHOD)

optimize-day2: build
	$(DOCKER_RUN) $(IMAGE) optimize --day 2 --method $(METHOD)

optimize-all: build
	$(DOCKER_RUN) $(IMAGE) optimize --day 1 --method $(METHOD)
	$(DOCKER_RUN) $(IMAGE) optimize --day 2 --method $(METHOD)

optimize-day%: build
	$(DOCKER_RUN) $(IMAGE) optimize --day $* --method $(METHOD)

# ── Utilities ────────────────────────────────────────────────────────
reset:
	rm -f entries/state.json

clean: reset
	docker rmi $(IMAGE) 2>/dev/null || true
