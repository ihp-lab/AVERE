
JSON_FOLDER="/wekafs/ict/achaubey/emotion_reasoning/data/videollava/train_json"
AUDIO_JSON_FOLDER="/wekafs/ict/achaubey/emotion_reasoning/audio_exp/data_preprocess/instruct_files"
IMAGE_FOLDER="/wekafs/ict/achaubey/emotion_reasoning/data/videollava"
VIDEO_FOLDER="/wekafs/ict/achaubey/emotion_reasoning/data/videollava"
AUDIO_FOLDER="/wekafs/ict/achaubey/emotion_reasoning/audio_exp/data"

CUDA_VISIBLE_DEVICES=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed --master_port=29501 videollava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./backbones/Video-LLaVA-7B \
    --version v1 \
    --data_path ${AUDIO_JSON_FOLDER}/giga_libri_pretrain.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower ./backbones/LanguageBind_Image \
    --video_folder ${VIDEO_FOLDER} \
    --video_tower ./backbones/LanguageBind_Video_merge \
    --audio_folder ${AUDIO_FOLDER} \
    --speech_tower ./backbones/whisper-large-v3 \
    --speech_projector_type qformer \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter False \
    --freeze_mm_mlp_adapter True \
    --tune_speech_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/audio_projector_pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 144 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 20000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048  --tokenizer_model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"
