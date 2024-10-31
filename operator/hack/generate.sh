#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
OPERATOR_GO_MODULE_ROOT="$(dirname "$SCRIPT_DIR")"
TOOLS_BIN_DIR="${SCRIPT_DIR}/tools/bin"

source "${TOOLS_BIN_DIR}/kube_codegen.sh"

function check_controller_gen_prereq() {
  if ! command -v controller-gen &>/dev/null; then
    echo >&2 "controller-gen is not available, cannot generate deepcopy/runtime.Object for the API types and cannot generate CRDs"
    exit 1
  fi
}

function generate_deepcopy_defaulter() {
  kube::codegen::gen_helpers \
    --boilerplate "${SCRIPT_DIR}/boilerplate.go.txt" \
    "${OPERATOR_GO_MODULE_ROOT}/api"
}

function generate_clientset() {
  kube::codegen::gen_client \
    --with-watch \
    --output-dir "${OPERATOR_GO_MODULE_ROOT}/client" \
    --output-pkg "github.com/NVIDIA/grove/operator/client" \
    --boilerplate "${SCRIPT_DIR}/boilerplate.go.txt" \
    "${OPERATOR_GO_MODULE_ROOT}/api"
}

function generate_crds() {
  local output_dir="${OPERATOR_GO_MODULE_ROOT}/config/crd/bases"
  local package="github.com/NVIDIA/grove/operator/api/podgangset/v1alpha1"
  local package_path="$(go list -f '{{.Dir}}' "${package}")"

  if [ -z "${package_path}" ]; then
    echo >&2 "Could not locate directory for package: ${package}"
    exit 1
  fi

  if [ -z "${output_dir}" ]; then
    mkdir -p "${output_dir}"
  fi

  # clean all generated crd files
  if ls "${output_dir}/*.yaml" 1> /dev/null 2>&1; then
    rm "${output_dir}/*.yaml"
  fi

  controller-gen crd paths="${package_path}" output:crd:dir="${output_dir}" output:stdout
}

function main() {
  echo "> Generate..."
  go generate "${OPERATOR_GO_MODULE_ROOT}/..."

  echo "> Generating DeepCopy and Defaulting functions..."
  generate_deepcopy_defaulter

  echo "> Generating ClientSet for PodGangSet API..."
  generate_clientset

  check_controller_gen_prereq
  echo "> Generate CRDs..."
  generate_crds
}

main
