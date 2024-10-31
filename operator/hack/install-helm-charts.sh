#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail


function check_prerequisites() {
  if ! command -v helm &> /dev/null; then
    echo "helm is not installed. Please install helm from https://helm.sh/docs/intro/install"
    exit 1
  fi
}

function verify_and_install_charts() {

}

