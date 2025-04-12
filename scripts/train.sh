cuda_devices=$1

CUDA_VISIBLE_DEVICES="${cuda_devices}" python main.py --global_model 'meta-llama/Llama-3.2-3B'\
      --data_path  "./data/dataset1" \
      --val_data_path "./data/dataset1/flan_test_200_selected_nstrict_1.jsonl" \
      --output_dir  '/home/scratch/haoyungw/fedit/fedavg-3B-hetlora/'\
      --num_communication_rounds 20 \
      --num_clients 8 \
      --prompt_template_name 'alpaca_short' \
      --client_selection_frac 1 \
      --federation_mode "hetero"