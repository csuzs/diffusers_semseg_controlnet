export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export HF_HOME="/project_workspace/uic19759/pretrained_model_catalog/.hugging_face_cache"
export OUTPUT_DIR="runs"
export REQUESTS_CA_BUNDLE="$(conticertifi cert)"
accelerate launch --config_file=/project_workspace/uic19759/diffusers_semseg_controlnet/examples/controlnet/config/accelerate_config_a100_single.yaml  \
 --multi_gpu train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --cache_dir=$HF_HOME \
 --train_data_dir='data' \
 --image_column "image" \
 --conditioning_image_column "condition" \
 --caption_column "caption" \
 --max_train_steps=126040 \
 --resolution=1024 \
 --learning_rate=1e-5 \
 --validation_image "/project_workspace/uic19759/bdd/bdd100k/images/10k/val/7d06fefd-f7be05a6.jpg" \
 --validation_prompt "Traffic scene. Snowy weather. Daytime. City street. High resolution."  \
 --mixed_precision="fp16" \
 --tracker_project_name="controlnet_semseg" \
 --train_batch_size=1 \
 --gradient_accumulation_steps=1 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --checkpointing_steps=2000 \
 --resume_from_checkpoint='latest' \
 --report_to='wandb' \
 --pretrained_vae_model_name_or_path='madebyollin/sdxl-vae-fp16-fix'