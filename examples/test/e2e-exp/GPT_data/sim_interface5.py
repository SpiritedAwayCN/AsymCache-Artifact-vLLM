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


vllm_host_ip = os.popen("hostname -I | awk '{print $1}'").read().strip()
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=150*60*60)
API_URL = f'http://{vllm_host_ip}:8040/v1/completions'
parser = argparse.ArgumentParser()
parser.add_argument('--frac', '-f', type=int, default=10)
parser.add_argument('--qps', '-q', type=float, default=1.00)
parser.add_argument('--model_size', '-ms', type=str, choices=['8B', '70B'], default='70B')
args = parser.parse_args()


input_file = "ShareGPT_outputs_gamma_qps100_CV0.25_70B_1.0.csv"
output_file_name = f"/jizhicfs/marvinhjia/sca/yilin/vllm-dev/examples/test/e2e-exp/GPT_results/ShareGPT_simulator_lru_8B_overall_qps1.00_frac100_CV0.25.jsonl"
df = pd.read_csv(input_file)
model = "/jizhicfs/marvinhjia/sca/models/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
tokenizer = AutoTokenizer.from_pretrained(model)

# load datasets
datasets = {}
with open("/jizhicfs/marvinhjia/sca/sharegpt_500.jsonl", 'r', encoding='utf-8') as file:
    file_name = "sharegpt"
    for line_number, line in enumerate(file):
        data = json.loads(line)
        if file_name not in datasets:
            datasets[file_name] = []
        datasets[file_name].append(data)


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

        msgs2 = []
        for idx, question in enumerate(data['instructions'][:turn_id + 1]):
            answer_text = data['outputs'][idx]
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