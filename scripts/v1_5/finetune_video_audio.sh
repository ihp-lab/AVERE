JSON_FOLDER="/wekafs/ict/achaubey/emotion_reasoning/data/videollava/train_json"
IMAGE_FOLDER="/wekafs/ict/achaubey/emotion_reasoning/data/videollava"
VIDEO_FOLDER="/wekafs/ict/achaubey/emotion_reasoning/audio_exp/data_video"
AUDIO_FOLDER="/wekafs/ict/achaubey/emotion_reasoning/audio_exp/data"
INSTRUCTION_DATA_FOLDER="/wekafs/ict/achaubey/emotion_reasoning/audio_exp/data/instruction_data"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed --master_port=29500 videollava/train/train_mem.py \
    --deepspeed ./scripts/zero2_offload.json \
    --model_name_or_path ./backbones/Video-LLaVA-7B \
    --version v1 \
    --data_path /wekafs/ict/achaubey/emotion_reasoning/audio_exp/data_preprocess/instruct_files/ferv39k_mafw_mer2025_single_descraw_datasets-naive_gemini_avlong-ferv39k_mafw_mer2025_single-gemini_qa-250words-modality_hallucination0.05_qa_extras_shuffled_choices.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower ./backbones/LanguageBind_Image \
    --video_folder ${VIDEO_FOLDER} \
    --video_tower ./backbones/LanguageBind_Video_merge \
    --mm_projector_type mlp2x_gelu \
    --audio_folder ${AUDIO_FOLDER} \
    --speech_tower ./backbones/whisper-large-v3 \
    --speech_projector_type qformer \
    --pretrain_speech_mlp_adapter ./checkpoints/audio_projector_pretrain/speech_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/avere_base \
    --num_train_epochs 1 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048  --tokenizer_model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"