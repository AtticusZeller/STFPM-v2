#!/usr/bin/env bash

set -e
set -x

bash scripts/format.sh
bash scripts/lint.sh
pytest tests
pre-commit run --all-files
