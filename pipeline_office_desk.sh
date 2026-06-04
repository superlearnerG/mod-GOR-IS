#!/bin/bash
set -uo pipefail

BASE_DATASET="../data"
# BASE_DATASET="../data/GOR-IS-datasets/gor-is-synthetic"


run_scene() {
    local scene="$1"
    local scene_path="${BASE_DATASET}/${scene}"
    local output_dir="../output/${scene}/mod-GOR-IS/5060000" # 命名格式：3月10日20点07分

    TORCH_CUDA_ARCH_LIST=8.9 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py --eval \
        -s "${scene_path}" \
        -m "${output_dir}" \
        --iterations 30000 \
        --specular_mask_id 255 \

    TORCH_CUDA_ARCH_LIST=8.9 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py --eval \
        -s "${scene_path}" \
        -m "${output_dir}" \
        --include_mask \
        --segmentation_load_iteration -1 \
        --specular_object_id 255 \
        --desk_object_id 255 \
        --finetune_mask \
        --enable_object_postprocess \
        --object_postprocess_voxel_scale 1.0 \
        --object_postprocess_dilation_voxels 2 \
        --object_postprocess_mask_thresh 0.0 

    TORCH_CUDA_ARCH_LIST=8.9 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python decouple_render.py \
        -s "${scene_path}" \
        -m "${output_dir}" \
        --render_isolated \
        --decouple_object_id 17 34 51 68 85 102 119 \
        --render_mode decouple

    TORCH_CUDA_ARCH_LIST=8.9 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python export_desk_atlas.py \
        -s "${scene_path}" \
        -m "${output_dir}" \
        --iteration -1 \
        --desk_object_id 255 \
        --desk_atlas_long_side 1254 \
        --desk_atlas_size_multiple 6 \
        --ccm_max_mask_samples 1000000 \
        --support_object_ids 17 34 51 68 85 102 119

    # python lama_inpaint_desk_atlas.py \
    #     --workroot_path "${output_dir}" \
    #     --mask_dilation 31  

    # TORCH_CUDA_ARCH_LIST=8.9 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python inpaint.py \
    #     -s "${scene_path}" \
    #     -m "${output_dir}" \
    #     --iteration 30000 \
    #     --iterations 35000 \
    #     --desk_object_id 255 \
    #     --decouple_object_id 17 34 51 68 85 102 119 136 153 170 187 204 \
    #     --lambda_normal 0.0 \
    #     --desk_support_shrink_px 15.0

    # TORCH_CUDA_ARCH_LIST=8.9 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python decouple_render.py \
    #     -s "${scene_path}" \
    #     -m "${output_dir}" \
    #     --iteration 35000 \
    #     --decouple_object_id 17 34 51 68 85 102 119 136 153 170 187 204 \
    #     --render_isolated \
    #     --render_mode decouple+inpaint

}

scenes=(
  office_desk
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