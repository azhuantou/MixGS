#!/bin/bash

# Adjust to the output folder name of your pruned model; following setting aligns with run_prune_finetune_my_scene.sh
SCENES=(my_scene_light_40 my_scene_light_50 my_scene_light_60)
VQ_RATIO=0.4
CODEBOOK_SIZE=8192

for SCENE in "${SCENES[@]}"   # Add more scenes as needed
do
    IMP_PATH=./output/${SCENE}
    INPUT_PLY_PATH=./output/${SCENE}_distill/point_cloud/iteration_40000/point_cloud.ply
    SAVE_PATH=./output/${SCENE}_vq

    CMD="CUDA_VISIBLE_DEVICES=0 python vectree/vectree.py \
    --important_score_npz_path ${IMP_PATH} \
    --input_path ${INPUT_PLY_PATH} \
    --save_path ${SAVE_PATH} \
    --vq_ratio ${VQ_RATIO} \
    --codebook_size ${CODEBOOK_SIZE} \
    "
    eval $CMD
done