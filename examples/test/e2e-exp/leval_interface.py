import time
import json
import random
import os
import glob
import sys
import argparse
import pandas as pd
import asyncio
import aiohttp
import tqdm

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

BASE_DIR = '/processing_data/infra/sca/dataset/LEval'
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
API_URL = 'http://localhost:8000/v1/completions'

parser = argparse.ArgumentParser()
parser.add_argument('--div', '-d', type=int, default=20)
parser.add_argument('--qps', '-q', type=float, default=1)
args = parser.parse_args()

output_file_name = f"outputs_q{args.qps}_d{args.div}.jsonl"

df = pd.read_csv(f'./leval_data/outputs_q{args.qps}_d{args.div}.csv')

# model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = "LargeWorldModel/LWM-Text-1M"
tokenizer = AutoTokenizer.from_pretrained(model)

# load datasets
datasets = {}
for jsonl_file in glob.glob(f"{BASE_DIR}/LEval-data/**/*.jsonl", recursive=True):
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file):
            data = json.loads(line)
            file_name = os.path.basename(jsonl_file)
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
    
def load_requests_from_csv(file_path):
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')  # 转换为日期时间格式
    return df

def prepare_requests():
    for _, row in df.iterrows():
        file_name, line_number, turn_id = row['Dataset Name'], row['Data Index'], row['Turn']
        data = datasets[file_name][line_number]

        msgs2 = [{'role': 'user', 'content': data['input'] + '\n\n'}]
        for idx, question in enumerate(data['instructions'][:turn_id + 1]):
            if idx == 0:
                msgs2[0]['content'] += question
                msgs2.append({'role': 'assistant', 'content': data['outputs'][idx]})
            else:
                msgs2.append({'role': 'user', 'content': question})
                msgs2.append({'role': 'assistant', 'content': data['outputs'][idx]})

        prompt, answer = generate_prompt_and_answer(msgs2)
        # print(f"prompt:\n{prompt}\n\n")
        # print(f"answer:\n{answer}\n\n")
        # 将prompt加入请求队列
        
        timestamp = row['Timestamp']
        yield timestamp, prompt, answer

async def fetch(prompt, answer, output_file):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": model,
            "prompt": prompt,
            "std_output_tokens": answer,
            "max_tokens": 4096,
        }
        headers = {
            "Content-Type": "application/json",
            # "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }
        
        async with session.post(API_URL, json=payload, headers=headers) as response:
            response_json = await response.json()
            # output_file is a .jsonl file
            output_file.write(json.dumps(response_json) + '\n')

async def schedule_request(output_file):
    first_time = -1
    tasks = []
    
    for timestamp, prompt, answer in tqdm.tqdm(prepare_requests(), total=len(df)):
        now = time.time()
        if first_time == -1:
            first_time = now
            now = 0
        else:
            now -= first_time
        
        if timestamp > now:
            await asyncio.sleep(timestamp - now)
        task = asyncio.create_task(fetch(prompt, answer, output_file))
        tasks.append(task)
    
    await asyncio.gather(*tasks)
        

if __name__ == '__main__':
    with open(output_file_name, 'w') as output_file:
        asyncio.run(schedule_request(output_file))