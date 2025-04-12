from peft import (
    set_peft_model_state_dict,
    get_peft_model,
    LoraConfig
)
import torch
import os
from torch.nn.functional import normalize


def FedAvg(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch):
    """
    Standard Federated Averaging for LoRA adapters with same rank.
    """
    weights_array = normalize(
        torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
                     dtype=torch.float32),
        p=1, dim=0)

    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id),
                                         "pytorch_model.bin")
        single_weights = torch.load(single_output_dir)
        if k == 0:
            weighted_single_weights = {key: single_weights[key] * (weights_array[k]) for key in
                                       single_weights.keys()}
        else:
            weighted_single_weights = {key: weighted_single_weights[key] + single_weights[key] * (weights_array[k])
                                       for key in
                                       single_weights.keys()}

    set_peft_model_state_dict(model, weighted_single_weights, "default")

    return model


def load_hetlora_weights(client_model, global_model, client_rank):
    """
    Load weights from the global model to a client model with a specific rank.
    This handles the case where ranks differ.
    """
    global_state = global_model.state_dict()
    client_state = client_model.state_dict()
    
    # Copy weights where possible
    for key in client_state.keys():
        if key in global_state:
            if "lora_A" in key:
                # For lora_A, we need to take a slice of the correct size
                # client: [client_rank, in_features]
                # global: [global_rank, in_features]
                if client_state[key].shape[0] != global_state[key].shape[0]:
                    # Only take the first client_rank rows
                    client_state[key] = global_state[key][:client_state[key].shape[0]].clone()
                else:
                    client_state[key] = global_state[key].clone()
            elif "lora_B" in key:
                # For lora_B, we need to take a slice of the correct size 
                # client: [out_features, client_rank]
                # global: [out_features, global_rank]
                if client_state[key].shape[1] != global_state[key].shape[1]:
                    # Only take the first client_rank columns
                    client_state[key] = global_state[key][:, :client_state[key].shape[1]].clone()
                else:
                    client_state[key] = global_state[key].clone()
            else:
                # For other weights, copy directly
                client_state[key] = global_state[key].clone()
    
    # Load the adjusted state into the client model
    client_model.load_state_dict(client_state)
    return client_model


def HetLoRA(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch, client_lora_ranks):
    """
    Heterogeneous LoRA aggregation that can handle different ranks across clients.
    """
    # Get the maximum rank used by any client
    max_rank = max(client_lora_ranks.values())
    
    # Normalize weights based on dataset sizes
    weights_array = normalize(
        torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
                     dtype=torch.float32),
        p=1, dim=0)
    
    # Dictionary to store aggregated weights
    aggregated_weights = {}
    
    # First, build a template of the expected aggregated weights structure
    # This is important to initialize all weights properly
    for client_id in selected_clients_set:
        client_weights_path = os.path.join(output_dir, str(epoch), f"local_output_{client_id}", "pytorch_model.bin")
        client_weights = torch.load(client_weights_path)
        
        # Initialize the aggregated weights structure based on the first client
        if not aggregated_weights:
            for key, weight in client_weights.items():
                if "lora_A" in key:
                    # For lora_A matrices: [rank, in_features] -> [max_rank, in_features]
                    in_features = weight.shape[1]
                    aggregated_weights[key] = torch.zeros((max_rank, in_features), 
                                                         dtype=weight.dtype,
                                                         device=weight.device)
                elif "lora_B" in key:
                    # For lora_B matrices: [out_features, rank] -> [out_features, max_rank]
                    out_features = weight.shape[0]
                    aggregated_weights[key] = torch.zeros((out_features, max_rank), 
                                                         dtype=weight.dtype,
                                                         device=weight.device)
                else:
                    # For other weights, just initialize with zeros of the same shape
                    aggregated_weights[key] = torch.zeros_like(weight)
        break  # We only need one client to initialize the structure
    
    # Now perform the aggregation
    for k, client_id in enumerate(selected_clients_set):
        client_weights_path = os.path.join(output_dir, str(epoch), f"local_output_{client_id}", "pytorch_model.bin")
        client_weights = torch.load(client_weights_path)
        client_rank = client_lora_ranks[client_id]
        
        for key, weight in client_weights.items():
            if "lora_A" in key:
                # For lora_A, place client's weights in the first client_rank rows
                aggregated_weights[key][:client_rank] += weight * weights_array[k]
            elif "lora_B" in key:
                # For lora_B, place client's weights in the first client_rank columns
                aggregated_weights[key][:, :client_rank] += weight * weights_array[k]
            else:
                # For other weights, simply perform weighted averaging
                aggregated_weights[key] += weight * weights_array[k]
    
    # Create a fresh model with the maximum rank
    base_model = model.base_model
    config = LoraConfig(
        r=max_rank,
        lora_alpha=model.peft_config['default'].lora_alpha,
        target_modules=model.peft_config['default'].target_modules,
        lora_dropout=model.peft_config['default'].lora_dropout,
        bias=model.peft_config['default'].bias,
        task_type=model.peft_config['default'].task_type,
    )
    
    # Create a new model with the maximum rank
    new_model = get_peft_model(base_model, config)
    
    # Load the aggregated weights into the new model
    set_peft_model_state_dict(new_model, aggregated_weights, "default")
    
    return new_model