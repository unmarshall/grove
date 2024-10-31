REPO_ROOT           := $(shell dirname "$(realpath $(lastword $(MAKEFILE_LIST)))")
HACK_DIR            := $(REPO_ROOT)/hack

# Include tools targets
include $(HACK_DIR)/tools.mk

.PHONY: build
build:
	@$(HACK_DIR)/build.sh

.PHONY: lint
lint: $(GOLANGCI_LINT)
	@$(GOLANGCI_LINT) run -c ../.golangci.yaml ./...

.PHONY: format
format: $(GOIMPORTS_REVISER)
	@$(HACK_DIR)/format.sh ./api/ ./internal/ ./cmd/

.PHONY: generate
generate: $(CODE_GENERATOR) $(CONTROLLER_GEN)
	@$(HACK_DIR)/generate.sh

# Make targets for local development and testing
# -------------------------------------------------------------
.PHONY: kind-up
kind-up: $(KIND) $(YQ)
	@$(HACK_DIR)/kind-up.sh

.PHONY: kind-down
kind-down: $(KIND)
	@$(HACK_DIR)/kind-down.sh