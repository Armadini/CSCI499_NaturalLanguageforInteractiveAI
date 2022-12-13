python3 train.py \
    --in_data_fn=lang_to_sem_data.json \
    --model_output_dir=experiments/s2s \
    --batch_size=10 \
    --num_epochs=100 \
    --val_every=5 \
    --force_cpu \
    --join_instructions