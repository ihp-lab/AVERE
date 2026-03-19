IMAGE_FOLDER="/put/any/dummy/path"
VIDEO_FOLDER="/wekafs/ict/achaubey/emotion_reasoning/audio_exp/data_video"
AUDIO_FOLDER="/wekafs/ict/achaubey/emotion_reasoning/audio_exp/data"
INSTRUCTION_DATA_FOLDER="/wekafs/ict/achaubey/emotion_reasoning/audio_exp/data/instruction_data"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed --master_port=29503 videollava/train/dpo_train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoints/avere_base \
    --dpo_alpha 1.0 --beta 0.1 --gamma 0 --tpd_gamma 0.2 --use_tpd True \
    --version v1 \
    --data_path /wekafs/ict/achaubey/emotion_reasoning/audio_exp/data_preprocess/dpo_files/mafw_mer-av_reasoning_modality_oversampled_classification-two_reject_only_one_irrelevant.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower ./backbones/LanguageBind_Image \
    --video_folder ${VIDEO_FOLDER} \
    --video_tower ./backbones/LanguageBind_Video_merge \
    --mm_projector_type mlp2x_gelu \
    --audio_folder ${AUDIO_FOLDER} \
    --speech_tower ./backbones/whisper-large-v3 \
    --speech_projector_type qformer \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints/avere_final \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --freeze_mm_mlp_adapter True \
    --freeze_speech_mlp_adapter True \
    --save_total_limit 50 \
    --learning_rate 5e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048  --tokenizer_model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"