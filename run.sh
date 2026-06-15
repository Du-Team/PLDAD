#!/bin/bash
set -e
GPU_ID=0
DIFF=linear
DLT=x0
RUN_IDS=(1 2 3)
CONFIGS=(
  "ASD 64 0.5 0"
  "MSL 4 0.3 0.5"
  "PSM 96 0.7 0.8"
  "SMAP 4 0.3 0.8"
  "SMD 8 0 0"
  "SWAT 4 0.7 0.5"
)
for cfg in "${CONFIGS[@]}"; do
  read -r ds window MASK JITTER <<< "${cfg}"
  EXP_TAG="m${MASK}_j${JITTER}_d${DIFF}"
  for RUN_ID in "${RUN_IDS[@]}"; do
    echo "===== [${ds}] Stage 1: MFAE | run=${RUN_ID} ====="
    python main.py --dataset "${ds}" \
                 --train_stage mfae \
                 --window_size ${window} \
                 --mask_ratio ${MASK} \
                 --jitter_scale ${JITTER} \
                 --diff_beta_schedule ${DIFF} \
                 --diffusion_loss_type ${DLT} \
                 --suffix "${EXP_TAG}" \
                 --run_number ${RUN_ID} \
                 --gpu_ids ${GPU_ID}
    echo "===== [${ds}] Stage 2: LPL | run=${RUN_ID} ====="
    python main.py --dataset "${ds}" \
                   --train_stage lpl \
                   --mask_ratio ${MASK} \
                   --jitter_scale ${JITTER} \
                   --diff_beta_schedule ${DIFF} \
                   --diffusion_loss_type ${DLT} \
                   --suffix "${EXP_TAG}" \
                   --run_number ${RUN_ID} \
                   --window_size ${window} \
                   --gpu_ids ${GPU_ID}
    echo "===== [${ds}] Stage 3: JFT | run=${RUN_ID} ====="
    python main.py --dataset "${ds}" \
                   --train_stage jft \
                   --mask_ratio ${MASK} \
                   --jitter_scale ${JITTER} \
                   --diff_beta_schedule ${DIFF} \
                   --diffusion_loss_type ${DLT} \
                   --suffix "${EXP_TAG}" \
                   --run_number ${RUN_ID} \
                   --window_size ${window} \
                   --gpu_ids ${GPU_ID}
  done
done
