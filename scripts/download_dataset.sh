#!/usr/bin/env bash

mkdir -p datasets
wget -c https://huggingface.co/datasets/Atticux/InsPLAD/resolve/main/InsPLAD-fault-unsupervised.zip?download=true -O datasets/InsPLAD-fault-unsupervised.zip -q --show-progress
unzip datasets/InsPLAD-fault-unsupervised.zip -d datasets
rm datasets/InsPLAD-fault-unsupervised.zip

# Ref: https://github.com/andreluizbvs/InsPLAD/tree/main?tab=readme-ov-file#insplad-inspection-of-power-line-assets-dataset
# dataset from https://drive.google.com/drive/folders/1psHiRyl7501YolnCcB8k55rTuAUcR9Ak?usp=drive_link, InsPLAD-fault/unsupervised_anomaly_detection.zip
