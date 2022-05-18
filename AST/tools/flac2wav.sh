#!/bin/bash

FLAC_DIR=$1
OUT_DIR=$2

for file in $FLAC_DIR/*; do
    if [[ ${file} == *".flac"* ]]; then
        fname=$(basename $file | cut -d'.' -f1)
        #ffmpeg -i ${file} ${OUT_DIR}/${fname}.wav
        #signal needs to be mono and 16khz !!!!!
        ffmpeg -i ${file}  -ac 1 -ar 16000  ${OUT_DIR}/${fname}.wav
    fi
done
