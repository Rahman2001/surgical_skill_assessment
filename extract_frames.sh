#!/bin/bash

if [ "$2" == "" ]; then
    echo "Please specify (1) the path to the dense flow build directory and (2) the path to the directory that contains the JIGSAWS data."
    exit 1
fi
DF_BUILD_DIR="$1"
JIGSAWS_ROOT="$2"

if [[ "$DF_BUILD_DIR" != */ ]]; then
    DF_BUILD_DIR="${DF_BUILD_DIR}/"
fi
if [[ "$JIGSAWS_ROOT" != */ ]]; then
    JIGSAWS_ROOT="${JIGSAWS_ROOT}/"
fi

STEP_SIZE="$3"
if [ "$STEP_SIZE" == "" ]; then
    STEP_SIZE=3
fi
NUM_GPU="$4"
if [ "$NUM_GPU" == "" ]; then
    NUM_GPU=1
fi
WORKERS_PER_GPU="$5"
if [ "$WORKERS_PER_GPU" == "" ]; then
    WORKERS_PER_GPU=4
fi
OUT_DIR="$6"
if [ "$OUT_DIR" == "" ]; then
    OUT_DIR="frames"
fi

echo "Will try to process data in ${JIGSAWS_ROOT} using step size ${STEP_SIZE}, ${NUM_GPU} GPU(s), and ${WORKERS_PER_GPU} worker(s) per GPU."

python3 ./build_of.py "${JIGSAWS_ROOT}Suturing/video" "${JIGSAWS_ROOT}Suturing/${OUT_DIR}" -s "$STEP_SIZE" --num_gpu "$NUM_GPU" --num_worker "$WORKERS_PER_GPU" \
--df_path "${DF_BUILD_DIR}" --new_width 340 --new_height 256 --suffix "_capture2.avi"

python3 ./build_of.py "${JIGSAWS_ROOT}Needle_Passing/video" "${JIGSAWS_ROOT}Needle_Passing/${OUT_DIR}" -s "$STEP_SIZE" --num_gpu "$NUM_GPU" --num_worker "$WORKERS_PER_GPU" \
--df_path "${DF_BUILD_DIR}" --new_width 340 --new_height 256 --suffix "_capture2.avi"

python3 ./build_of.py "${JIGSAWS_ROOT}Knot_Tying/video" "${JIGSAWS_ROOT}Knot_Tying/${OUT_DIR}" -s "$STEP_SIZE" --num_gpu "$NUM_GPU" --num_worker "$WORKERS_PER_GPU" \
--df_path "${DF_BUILD_DIR}" --new_width 340 --new_height 256 --suffix "_capture2.avi"
