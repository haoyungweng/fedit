import os
import fire
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from peft import (
    LoraConfig, 
    PeftModel,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    GenerationConfig,
    set_seed,
)
from utils.prompter import Prompter
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class EvalDataset(Dataset):
    def __init__(self, file, prompter, tokenizer):
        self.prompter = prompter
        self.tokenizer = tokenizer
        with open(file, 'r') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        line = line.strip()
        ques = json.loads(line)
        sample = ques['instruction']
        prompt = self.prompter.generate_prompt(
            ques['instruction'],
            ques["input"] if 'input' in ques.keys() else None,
        )
        return prompt, sample


def generate(
    base_model: str = "",
    output_dir: str = './lora-shepherd/',
    fedavg: bool = False,
    client_id: int = 0,
    communication_rounds: int = 50,
    prompt_template_name: str = "",  # The prompt template to use, will default to alpaca.
    test_file_path: str="",
    save_dir: str="",
    batch_size: int=2,
):
    set_seed(42)
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    prompter = Prompter(prompt_template_name)
    gpu_count = torch.cuda.device_count()
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    if fedavg: # global model using fedavg
        lora_config_path = output_dir
        lora_weights_path = os.path.join(output_dir, str(communication_rounds), "adapter_model.bin")   
    else:
        lora_config_path = os.path.join(output_dir, str(communication_rounds), f"local_output_{client_id}")
        lora_weights_path = os.path.join(lora_config_path, "pytorch_model.bin")
    config = LoraConfig.from_pretrained(lora_config_path)
    if gpu_count < 3:
        print(gpu_count)
        lora_weights = torch.load(lora_weights_path, map_location=lambda storage, loc: storage.cuda(0))
    else:
        lora_weights = torch.load(lora_weights_path)
    
    model = PeftModel(model, config)
    set_peft_model_state_dict(model, lora_weights, "default")
    del lora_weights


    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()


    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=50,
        num_beams=1,
        max_new_tokens=128,
        input_ids=None,
        attention_mask=None,
        **kwargs,
    ):
        if input_ids is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
        else:
            prompt = prompter.generate_prompt(instruction, input)
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

        generation_config = GenerationConfig(
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_beams=num_beams,
            num_return_sequences=1,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.1,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        if len(generation_output.sequences) ==1:
            s = generation_output.sequences[0]
            output = tokenizer.decode(s)
            ans = prompter.get_response(output).split(tokenizer.eos_token)[0]
        else:
            s = generation_output.sequences.cpu()
            output = tokenizer.batch_decode(s)
            ans = [prompter.get_response(t).split(tokenizer.eos_token)[0] for t in output]
        return ans


    eval_dataset = EvalDataset(test_file_path, prompter, tokenizer)
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    all_res = []
    for prompts, text in tqdm(dataloader):
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        res = evaluate(None, input_ids=input_ids, attention_mask=attention_mask)
        all_res.extend(res)
    

    save_dir = os.path.join(save_dir, str(communication_rounds))
    os.makedirs(save_dir, exist_ok=True)
    save_file_path = os.path.join(save_dir, f"client_{client_id}_output.jsonl")

    lines = open(test_file_path).readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        ques = json.loads(line)
        res = all_res[i]

        tmp = {}
        tmp['text'] = ques['instruction']
        tmp['answer'] = res
        tmp['category'] = ques['category']

        with open(save_file_path, 'a+', encoding='utf-8') as f:
            json.dump(tmp, f, ensure_ascii=False)
            f.write('\n')  
        
        print('num:', i+1)
        print("Instruction:", tmp['text'])
        print("Response:", tmp['answer'])
        print("*****************************************************")


if __name__ == "__main__":
    fire.Fire(generate)