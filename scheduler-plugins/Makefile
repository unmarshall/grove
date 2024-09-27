REPO_ROOT           := $(shell dirname "$(realpath $(lastword $(MAKEFILE_LIST)))")
HACK_DIR            := $(REPO_ROOT)/hack

include $(HACK_DIR)/tools.mk

KUBERNETES_VERSION=$(call version_gomod,k8s.io/kubernetes)
SCHEDULER_DEV_VERSION=$(KUBERNETES_VERSION)-dev

.PHONY: lint
lint: $(GOLANGCI_LINT)
	@$(GOLANGCI_LINT) run -c ../.golangci.yaml ./...

.PHONY: build
build:
	@go build -ldflags '-X k8s.io/component-base/version.gitVersion=$(SCHEDULER_DEV_VERSION)' -o bin/kube-scheduler cmd/main.go
