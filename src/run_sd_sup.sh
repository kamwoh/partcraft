python train_dreamcreature_sd.py \
     --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
     --train_data_dir=data/cub200_2011 \
     --resolution=512 --random_flip --train_batch_size=2 --gradient_accumulation_steps=4 \
     --num_train_epochs=100 --checkpointing_steps=749 --learning_rate=0.0001 \
     --lr_scheduler="constant" --lr_warmup_steps=0 --seed=42 --output_dir="sd15-cub200-sup" \
     --validation_prompt="a photo of a 0:16 1:16 2:16 4:16 6:16" \
     --num_validation_images 8 --num_parts 8 --num_k_per_part 256 --filename="train.txt" \
     --code_filename="train_caps_better_m8_k256.txt" --projection_nlayers=1 \
     --use_templates --vector_shuffle --snr_gamma=5 \
     --attn_loss=0.01 --use_gt_label --bg_code=7 \
     --resume_from_checkpoint="latest" --mixed_precision="fp16"
