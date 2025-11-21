import os
import glob
import time
import json
import tqdm
import asyncio
import aiohttp
import argparse
import pandas as pd
from transformers import AutoTokenizer

BASE_DIR = '/jizhicfs/marvinhjia/sca'
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=150*60*60)
vllm_host_ip = os.popen("hostname -I | awk '{print $1}'").read().strip()
API_URL = f'http://{vllm_host_ip}:8048/v1/completions'

parser = argparse.ArgumentParser()
parser.add_argument('--div', '-d', type=int, default=30)
parser.add_argument('--qps', '-q', type=float, default=0.02)
# parser.add_argument('--half_life', '-hl', type=int, default=40)
parser.add_argument('--policy', '-p', type=str, choices=['mwf', 'lru'], default='mwf')
parser.add_argument('--dataset', '-ds', type=str, choices=['leval', 'longbench', 'LooGLE'], default='leval')
parser.add_argument('--use_model_output', '-mo', action='store_true', help='是否要启用模型生成的输出')
parser.add_argument('--model_output_file', '-mf', type=str, default=None, help='deepseek生成的输出jsonl文件路径')
args = parser.parse_args()

# 根据dataset参数选择路径前缀
if args.dataset == 'leval':
    dataset_dir = f"{BASE_DIR}/Four-LEval-data"
    dataset_prefix = "LEval"
elif args.dataset == 'longbench':
    dataset_dir = f"{BASE_DIR}/Longbench"
    dataset_prefix = "LongBench"
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")


input_file_name = f'./{dataset_prefix.lower()}_data/1_outputs_q{args.qps}_d{args.div}.csv'
output_file_name = f"./{dataset_prefix}_simulator_{args.policy}_qps{args.qps}_div{args.div}_P90.jsonl"
df = pd.read_csv(input_file_name)
model = "LargeWorldModel/LWM-Text-1M"
tokenizer = AutoTokenizer.from_pretrained(model)

# load datasets
datasets = {}
for jsonl_file in glob.glob(f"{dataset_dir}/**/*.jsonl", recursive=True):
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file):
            data = json.loads(line)
            file_name = os.path.basename(jsonl_file)
            if file_name not in datasets:
                datasets[file_name] = []
            datasets[file_name].append(data)    

# 加载deepseek的输出
model_outputs = {}
if args.use_model_output and args.model_output_file:
    with open(args.model_output_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            key = (item['file'], item['line'], item['turn'])
            model_outputs[key] = item['generate_text']
    print(f"[INFO] Loaded {len(model_outputs)} model-generated outputs from {args.model_output_file}")


def generate_prompt_and_answer(conversations):
    prompt, answer_token_ids = None, None
    if conversations[-1]['role'] == 'assistant':
        prompt = tokenizer.apply_chat_template(conversations[:-1], tokenize=False, add_generation_prompt=True)
        prompt_with_answer = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
        encoded_prompt = tokenizer.encode(prompt)
        encoded_prompt_with_answer = tokenizer.encode(prompt_with_answer)
        answer_token_ids = encoded_prompt_with_answer[len(encoded_prompt): ]
    return prompt, answer_token_ids


def prepare_requests():
    for _, row in df.iterrows():
        file_name, line_number, turn_id = row['Dataset Name'], row['Data Index'], row['Turn']
        data = datasets[file_name][line_number]

        msgs2 = [{'role': 'user', 'content': data['input'] + '\n\n'}]
        for idx, question in enumerate(data['instructions'][:turn_id + 1]):
            if args.use_model_output:
                gen_key = (file_name, line_number, idx)
                answer_text = model_outputs.get(gen_key, data['outputs'][idx])
            else:
                answer_text = data['outputs'][idx]
            if idx == 0:
                msgs2[0]['content'] += question
                msgs2.append({'role': 'assistant', 'content': answer_text})            
            else:
                msgs2.append({'role': 'user', 'content': question})
                msgs2.append({'role': 'assistant', 'content': answer_text})

        prompt, answer = generate_prompt_and_answer(msgs2)
        timestamp = row['Timestamp']
        yield timestamp, prompt, answer, file_name, line_number, turn_id


async def fetch(prompt, answer, output_file, file_name, line_number, turn_id):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": model,
            "prompt": prompt,
            "std_output_tokens": answer,
            "max_tokens": 16000,
        }
        headers = {
            "Content-Type": "application/json",
        }
        
        async with session.post(API_URL, json=payload, headers=headers) as response:
            response_json = await response.json()
            response_json.update({
                "file": file_name,
                "line": line_number,
                "turn": turn_id
            })
            output_file.write(json.dumps(response_json) + '\n')
        
        output_file.flush()


async def schedule_request(output_file):
    tasks = []
    first_time = -1
    
    for timestamp, prompt, answer, file_name, line_number, turn_id in tqdm.tqdm(prepare_requests(), total=len(df)):
        now = time.time()
        if first_time == -1:
            first_time = now
            now = 0
        else:
            now -= first_time
        
        if timestamp > now:
            await asyncio.sleep(timestamp - now)
        task = asyncio.create_task(fetch(prompt, answer, output_file, file_name, line_number, turn_id))
        tasks.append(task)
    
    await asyncio.gather(*tasks)


if __name__ == '__main__':
    with open(output_file_name, 'a') as output_file:
        asyncio.run(schedule_request(output_file))