export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export HF_HOME="/project_workspace/uic19759/pretrained_model_catalog/.hugging_face_cache"
export OUTPUT_DIR="infer_out"
export REQUESTS_CA_BUNDLE="$(conticertifi cert)"

python infer_sdxl.py