python metric.py \
    --target_file data/dataset1/flan_test_200_selected_nstrict_1.jsonl \
    --target_key output \
    --prediction_file outputs/fedavg-3B/20/19/client_0_output.jsonl \
    --prediction_key answer \
    --output_dir results/fedavg-3B/20/ \