#!/usr/bin/env bash

set -e
set -x

bash scripts/format.sh
bash scripts/lint.sh
pytest tests/test_transform.py tests/test_data.py -m "not slow"
pre-commit run --all-files
