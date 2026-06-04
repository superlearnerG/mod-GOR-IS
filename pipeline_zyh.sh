#!/bin/bash
set -uo pipefail

BASE_DATASET="../data"
# BASE_DATASET="../data/GOR-IS-datasets/gor-is-synthetic"


run_scene() {
    local scene="$1"
    local scene_path="${BASE_DATASET}/${scene}"
    local output_dir="../output/${scene}/mod-GOR-IS/4201800" # 命名格式：3月10日20点07分

    # TORCH_CUDA_ARCH_LIST=8.9 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py --eval \
    #     -s "${scene_path}" \
    #     -m "${output_dir}" \
    #     --iterations 30000 \
    #     --specular_mask_id 255 \

    # TORCH_CUDA_ARCH_LIST=8.9 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py --eval \
    #     -s "${scene_path}" \
    #     -m "${output_dir}" \
    #     --include_mask \
    #     --segmentation_load_iteration -1 \
    #     --specular_object_id 255 \
    #     --desk_object_id 255
    #     # --finetune_mask \
    #     # --enable_object_postprocess \
    #     # --object_postprocess_voxel_scale 2.0 \
    #     # --object_postprocess_dilation_voxels 2 \
    #     # --object_postprocess_mask_thresh 0.05 

    # TORCH_CUDA_ARCH_LIST=8.9 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python decouple_render.py \
    #     -s "${scene_path}" \
    #     -m "${output_dir}" \
    #     --iteration 30000 \
    #     --render_isolated \
    #     --decouple_object_id 12 25 38 51 63 76 \
    #     --render_mode decouple \
    #     --transparent_background

    TORCH_CUDA_ARCH_LIST=8.9 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python export_desk_atlas.py \
        -s "${scene_path}" \
        -m "${output_dir}" \
        --iteration -1 \
        --desk_atlas_long_side 1024 \
        --desk_atlas_size_multiple 8 \
        --ccm_max_mask_samples 1000000 \
        --desk_object_id 255 

    # python lama_inpaint_desk_atlas.py \
    #     --workroot_path "${output_dir}" \
    #     --mask_dilation 31  

    TORCH_CUDA_ARCH_LIST=8.9 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python inpaint.py \
        -s "${scene_path}" \
        -m "${output_dir}" \
        --iteration 30000 \
        --iterations 32000 \
        --desk_object_id 255 \
        --decouple_object_id 12 25 38 51 63 76 \
        --lambda_normal 0.0 \
        --reflection_visible \
        --desk_support_shrink_px 8 \
        --save_training_vis

    TORCH_CUDA_ARCH_LIST=8.9 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python decouple_render.py \
        -s "${scene_path}" \
        -m "${output_dir}" \
        --iteration 32000 \
        --decouple_object_id 12 25 38 51 63 76 \
        --render_isolated \
        --render_mode decouple+inpaint \
        --only_background
    #     --transparent_background
}

scenes=(
  scene_5_colmap 
)

failures=()

for scene in "${scenes[@]}"; do
  if run_scene "${scene}"; then
    echo "Finished ${scene}"
  else
    echo "Failed ${scene}, continuing to next scene"
    failures+=("${scene}")
  fi
done

if ((${#failures[@]})); then
  echo "Scenes failed: ${failures[*]}"
  exit 1
fi