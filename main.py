import os
import random
from typing import List
from tqdm import tqdm
import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from fed_utils import FedAvg, HetLoRA, load_hetlora_weights, client_selection, global_evaluation, GeneralClient
import datasets
from utils.prompter import Prompter

datasets.utils.logging.set_verbosity_error()


def fl_finetune(
        # model/data params
        global_model: str = '',
        data_path: str = './data',
        output_dir: str = './lora-shepherd/',
        # FL hyperparamas
        client_selection_strategy: str = 'random',
        client_selection_frac: float = 0.1,
        num_communication_rounds: int = 50,
        num_clients: int = 10,
        # Federation mode
        federation_mode: str = "homo",  # Can be "none", "homo", or "hetero"
        # Local training hyperparams
        local_batch_size: int = 128,  # 64,
        local_micro_batch_size: int = 8,
        local_num_epochs: int = 3,
        local_learning_rate: float = 3e-4,
        local_val_set_size: int = 0,
        val_data_path: str = "",
        local_save_steps: int = 3,
        cutoff_len: int = 512,
        # LoRA hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ],
        # llm hyperparams
        train_on_inputs: bool = False,
        group_by_length: bool = False,
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    # Validate federation mode
    assert federation_mode in ["none", "homo", "hetero"], \
        "federation_mode must be one of 'none' (no federation), 'homo' (homogeneous FedAvg), or 'hetero' (heterogeneous HetLoRA)"

    use_hetlora = (federation_mode == "hetero")
    use_federation = (federation_mode in ["homo", "hetero"])

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Federated Finetuning LLM-LoRA with params:\n"
            f"global_model: {global_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"federation_mode: {federation_mode}\n"
            f"client_selection_strategy: {client_selection_strategy}\n"
            f"client_selection_frac: {client_selection_frac}\n"
            f"num_communication_rounds: {num_communication_rounds}\n"
            f"num_clients: {num_clients}\n"
            f"local_batch_size: {local_batch_size}\n"
            f"local_micro_batch_size: {local_micro_batch_size}\n"
            f"local_num_epochs: {local_num_epochs}\n"
            f"local_learning_rate: {local_learning_rate}\n"
            f"local_val_set_size: {local_val_set_size}\n"
            f"local_save_steps: {local_save_steps}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        global_model
    ), "Please specify a --global_model, e.g. --global_modell='decapoda-research/llama-7b-hf'"

    data_path = os.path.join(data_path, str(num_clients))
    assert os.path.exists(data_path), "Please generate the data files for each client"

    # set up the global model & toknizer
    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    prompter = Prompter(prompt_template_name)
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        global_model,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(global_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"] if 'input' in data_point.keys() else None,
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"] if 'input' in data_point.keys() else None,
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    # If using HetLoRA, prepare client_lora_ranks dictionary
    client_lora_ranks = {}
    if use_hetlora:
        random.seed(42)  # For reproducibility
        for client_id in range(num_clients):
            # Randomly assign a rank from {4, 8, 16}
            client_lora_ranks[client_id] = random.choice([4, 8, 16])
        print("Using HetLoRA with client ranks:", client_lora_ranks)
        # Use the maximum rank for the global model
        global_lora_r = max(client_lora_ranks.values())
    else:
        # Use the same rank for all clients when using FedAvg or no federation
        global_lora_r = lora_r
        for client_id in range(num_clients):
            client_lora_ranks[client_id] = lora_r

    # Initialize the global model with the global LoRA configuration
    base_model = prepare_model_for_kbit_training(base_model)
    config = LoraConfig(
        r=global_lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, config)
    
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    print(f"The process of {'federated' if use_federation else 'isolated'} instruction-tuning has started..")
    previously_selected_clients_set = set()
    last_client_id = None
    local_dataset_len_dict = dict()
    output_dir = os.path.join(output_dir, str(num_clients))

    for epoch in tqdm(range(num_communication_rounds)):
        print("\nConducting the client selection")
        selected_clients_set = client_selection(num_clients, client_selection_frac, client_selection_strategy,
                                                other_info=epoch)

        for client_id in selected_clients_set:
            # If using heterogeneous LoRA or no federation, create a client-specific model
            if use_hetlora or not use_federation:
                client_lora_r = client_lora_ranks[client_id]
                print(f"Creating model for client {client_id} with rank {client_lora_r}")
                
                # Create a new client-specific configuration
                client_config = LoraConfig(
                    r=client_lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                
                # Create a new model with the client's specific rank
                client_model = get_peft_model(base_model, client_config)
                
                # If federation is enabled and not the first epoch, load weights from the global model where possible
                if use_federation and epoch > 0:
                    try:
                        client_model = load_hetlora_weights(client_model, model, client_lora_r)
                        print(f"Successfully loaded weights for client {client_id}")
                    except Exception as e:
                        print(f"Error loading weights for client {client_id}: {e}")
                        # Continue with fresh weights if there's an error
                
                # Use the client-specific model
                client = GeneralClient(client_id, client_model, data_path, output_dir)
            else:
                # For homogeneous federation (FedAvg), all clients use the same model
                client = GeneralClient(client_id, model, data_path, output_dir)

            print("\nPreparing the local dataset and trainer for Client_{}".format(client_id))
            client.preprare_local_dataset(generate_and_tokenize_prompt, local_val_set_size)
            client.build_local_trainer(tokenizer,
                                       local_micro_batch_size,
                                       gradient_accumulation_steps,
                                       local_num_epochs,
                                       local_learning_rate,
                                       group_by_length,
                                       ddp)

            print("Initiating the local training of Client_{}".format(client_id))
            client.initiate_local_training()

            print("Local training starts ... ")
            client.train()

            print("\nTerminating the local training of Client_{}".format(client_id))
            if use_hetlora or not use_federation:
                # For HetLoRA or no federation, we need to update the client_model that was created specifically for this client
                client_model, local_dataset_len_dict, previously_selected_clients_set, last_client_id = client.terminate_local_training(
                    epoch, local_dataset_len_dict, previously_selected_clients_set)
            else:
                # For homogeneous federation (FedAvg), we update the global model directly
                model, local_dataset_len_dict, previously_selected_clients_set, last_client_id = client.terminate_local_training(
                    epoch, local_dataset_len_dict, previously_selected_clients_set)
            
            del client

        # Only perform aggregation if federation is enabled
        if use_federation:
            print("Collecting the weights of clients and performing aggregation")
            if use_hetlora:
                model = HetLoRA(model,
                            selected_clients_set,
                            output_dir,
                            local_dataset_len_dict,
                            epoch,
                            client_lora_ranks)
            else:
                model = FedAvg(model,
                            selected_clients_set,
                            output_dir,
                            local_dataset_len_dict,
                            epoch)
            
            # Save the global model
            torch.save(model.state_dict(), os.path.join(output_dir, str(epoch), "adapter_model.bin"))
            config.save_pretrained(output_dir)
        else:
            # When no federation is used, we still want to save individual client models for evaluation
            print("No federation: Saving individual client models only")
            # No global model to save in this case

        # Please design the evaluation method based on your specific requirements in the fed_utils/evaluation.py file.
        # eval_loss = global_evaluation(model, val_data_path, generate_and_tokenize_prompt, 1, 'cuda')
        # print('communication round: ', epoch, ' the eval loss: ', eval_loss)


if __name__ == "__main__":
    fire.Fire(fl_finetune)