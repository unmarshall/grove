#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

function check_controller_gen_prereq() {
  if ! command -v controller-gen &>/dev/null; then
    echo >&2 "controller-gen is not available, cannot generate deepcopy/runtime.Object for the API types and cannot generate CRDs"
    exit 1
  fi
}

function generate_deepcopy() {
  cd "${PROJECT_DIR}/api" &&
    controller-gen "object:headerFile=${SCRIPT_DIR}/boilerplate.go.txt,year=2024" paths="./..."
}

function generate_crds() {
  local output_dir="${PROJECT_DIR}/config/crd/bases"
  local package="github.com/NVIDIA/grove/operator/api/v1alpha1"
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
  go generate "${PROJECT_DIR}/..."

  check_controller_gen_prereq

  echo "> Generate deepcopy/runtime.Object for API types..."
  generate_deepcopy

  echo "> Generate CRDs..."
  generate_crds
}

main
