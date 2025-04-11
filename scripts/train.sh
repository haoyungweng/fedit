cuda_devices=$1

CUDA_VISIBLE_DEVICES="${cuda_devices}" python main.py --global_model 'meta-llama/Llama-3.2-1B'\
      --data_path  "./data" \
      --output_dir  '/home/scratch/haoyungw/fedit/'\
      --num_communication_rounds 10 \
      --num_clients  10 \
      --train_on_inputs \
      --group_by_length