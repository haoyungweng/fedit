cuda_devices=$1

CUDA_VISIBLE_DEVICES="${cuda_devices}" python generate.py \
    --base_model meta-llama/Llama-3.2-3B \
    --output_dir /home/scratch/haoyungw/fedit/fedavg-3B/8/ \
    --communication_rounds 19 \
    --prompt_template_name alpaca_short \
    --test_file_path "./data/dataset1/flan_test_200_selected_nstrict_1.jsonl" \
    --save_dir ./outputs/fedavg-3B/20/ \
    --batch_size 4 \
    --fedavg