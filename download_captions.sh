#!/bin/bash

set -e

if [ -z "$1" ]; then
  echo "usage download_captions.sh [data dir]"
  #exit
fi

if [ "$(uname)" == "Darwin" ]; then
  UNZIP="tar -xf"
else
  UNZIP="unzip -nq"
fi

# Create the output directories.
OUTPUT_DIR="${1%/}"
SCRATCH_DIR="${OUTPUT_DIR}/raw-data"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SCRATCH_DIR}"
CURRENT_DIR=$(pwd)
WORK_DIR="$0.runfiles/im2txt/im2txt"

# Helper function to download and unpack a .zip file.
function download_and_unzip() {
  local BASE_URL=${1}
  local FILENAME=${2}

  if [ ! -f ${FILENAME} ]; then
    echo "Downloading ${FILENAME} to $(pwd)"
    wget -nd -c "${BASE_URL}/${FILENAME}"
  else
    echo "Skipping download of ${FILENAME}"
  fi
  echo "Unzipping ${FILENAME}"
  ${UNZIP} ${FILENAME}
}

cd ${SCRATCH_DIR}

# Download the captions.
BASE_URL="http://msvocds.blob.core.windows.net/annotations-1-0-3"
CAPTIONS_FILE="captions_train-val2014.zip"
INSTANCES_FILE="instances_train-val2014.zip"

download_and_unzip ${BASE_URL} ${CAPTIONS_FILE}
TRAIN_CAPTIONS_FILE="${SCRATCH_DIR}/annotations/captions_train2014.json"
VAL_CAPTIONS_FILE="${SCRATCH_DIR}/annotations/captions_val2014.json"

download_and_unzip ${BASE_URL} ${INSTANCES_FILE}
TRAIN_OBJ_FILE="${SCRATCH_DIR}/annotations/instances_train2014.json"
VAL_OBJ_FILE="${SCRATCH_DIR}/annotations/instances_val2014.json"

