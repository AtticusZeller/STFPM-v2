#!/usr/bin/env bash

# raw
python -m expt.main -c config/raw_resnet18.yml --train
python -m expt.main -c config/wideresnet.yml --train

# new
python -m expt.main -c config/resnet18.yml --train
python -m expt.main -c config/resnest101e.yml --train
python -m expt.main -c config/seresnext.yml --train