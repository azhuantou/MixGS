#!/bin/bash

# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

# Initial port number
port=6526

# Datasets
declare -a run_args=(
    "building_c20_r4_light_40"
    "building_c20_r4_light_50"
    "building_c20_r4_light_60"
)

# activate psudo view, else using train view for distillation 
declare -a virtue_view_arg=(
  "--augmented_view"
)

for arg in "${run_args[@]}"; do
  for view in "${virtue_view_arg[@]}"; do
    # Wait for an available GPU
    while true; do
      gpu_id=$(get_available_gpu)
      if [[ -n $gpu_id ]]; then
        echo "GPU $gpu_id is available. Starting distill_train.py with dataset '$arg' and options '$view' on port $port"
        CUDA_VISIBLE_DEVICES=$gpu_id nohup python large_distill_train.py \
          -s "data/mill19/building-pixsfm/train" \
          -m "output/${arg}_distill" \
          --start_checkpoint "output/${arg}/chkpnt30000.pth" \
          --iteration 40000 \
          --teacher_model "output/${arg}/chkpnt30000.pth" \
          --new_max_sh 2 \
          --position_lr_init 0.0000003 \
          --position_lr_final 0.000000003 \
          --scaling_lr 0.000075 \
          --position_lr_max_steps 40000 \
          --enable_covariance \
          $view \
          --port $port > "logs_prune/distill_${arg}_${view}.log" 2>&1 &

        # Increment the port number for the next run
        ((port++))
        # Allow some time for the process to initialize and potentially use GPU memory
        sleep 120
        break
      else
        echo "No GPU available at the moment. Retrying in 2 minute."
        sleep 120
      fi
    done
  done
done
wait
echo "All distill_train.py runs completed."
