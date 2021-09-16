#!/bin/bash

FLAC_DIR=$1
OUT_DIR=$2

for file in $FLAC_DIR/*; do
    if [[ ${file} == *".flac"* ]]; then
        fname=$(basename $file | cut -d'.' -f1)
        ffmpeg -i ${file} ${OUT_DIR}/${fname}.wav
    fi
done
