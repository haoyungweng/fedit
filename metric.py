from typing import Dict, List
import evaluate
import tqdm
import json
import fire
import os

def read_list(file,k):
    dic={}
    lines=open(file).readlines()
    for line in lines:
        line = line.strip()
        data = json.loads(line)
        if data['category'] not in dic.keys():
            dic[data['category']] = []
        tmpd=data[k]
        if data[k].endswith('</s>'):
            tmpd = data[k].split('</s>')[0]
        #if data['category'] in ['paraphrase','question_classification']:
           # tmpd = tmpd.split(' ')[0].strip(',')
        instruction_name = "instruction" if "instruction" in data else "text"
        dic[data['category']].append({"instruction": data[instruction_name], "output": tmpd})
    return dic

def rouge(targets, predictions):
    # results = metrics.rouge(targets, predictions)
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=predictions, references=targets)
    return results

def get_result(targets, predictions, save):
    results = {}
    total_target=[]
    total_pre=[]
    for k in targets.keys():
        target_list = []
        prediction_list = []
        for i, target in enumerate(targets[k]):
            assert target['instruction'] == predictions[k][i]['instruction'], "Instruction mismatch!"
            target_list.append(target['output'])
            prediction_list.append(predictions[k][i]['output'])

        result = rouge(targets=target_list, predictions=prediction_list)
        results[k] = result
        total_target.extend(target_list)
        total_pre.extend(prediction_list)
    results['total'] = rouge(total_target, total_pre)
    print(results)
    with open(save, 'w') as f:
        f.write(json.dumps(results))

def main(
      target_file: str = 'targets.jsonl',
      target_key: str = 'output',
      prediction_file: str = 'predictions.jsonl', 
      prediction_key: str = 'answer',
      output_dir: str = '',   
):
    os.makedirs(output_dir, exist_ok=True)
    file_name = prediction_file.split('/')[-1]
    targets = read_list(file=target_file, k=target_key)
    predictions = read_list(file=prediction_file, k=prediction_key)
    get_result(targets, predictions, os.path.join(output_dir, file_name.replace('.jsonl', '_rouge.json')))


if __name__ == "__main__":
   fire.Fire(main)