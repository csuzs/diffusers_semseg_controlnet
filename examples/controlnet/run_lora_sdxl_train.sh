export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export HF_HOME="/project_workspace/uic19759/pretrained_model_catalog/.hugging_face_cache"
export OUTPUT_DIR="lora_runs"
export REQUESTS_CA_BUNDLE="$(conticertifi cert)"
accelerate launch --config_file=/project_workspace/uic19759/diffusers_semseg_controlnet/examples/controlnet/config/accelerate_config_a100_single.yaml \
 --multi_gpu \
 --num_processes=4 \
 train_dreambooth_lora_sdxl_bdd.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --cache_dir=$HF_HOME \
 --train_data_dir="data" \
 --dataset_name="bdd" \
 --image_column="image" \
 --caption_column="caption" \
 --max_train_steps=126040 \
 --resolution=1024 \
 --learning_rate=1e-5 \
 --mixed_precision="fp16" \
 --train_batch_size=4 \
 --gradient_accumulation_steps=1 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --checkpointing_steps=250 \
 --resume_from_checkpoint='latest' \
 --report_to='wandb' \
 --pretrained_vae_model_name_or_path='madebyollin/sdxl-vae-fp16-fix'