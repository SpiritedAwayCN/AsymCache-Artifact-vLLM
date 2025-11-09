import os
import re
import glob
import json
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


BASE_DIR = "/jizhicfs/marvinhjia/sca/Longbench"
CSV_PATH = "/jizhicfs/marvinhjia/sca/data_preprocess/seqlen_stat_Longbench.csv"
OUTPUT_DIR = "/jizhicfs/marvinhjia/sca/data_preprocess"
OUTPUT_JSONL = os.path.join(OUTPUT_DIR, "longbench_deepseek_results.jsonl")

TENSOR_PARALLEL_SIZE = 8
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
SAMPLING_PARAMS = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=8192)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-70B")

datasets = {}
for jsonl_file in glob.glob(f"{BASE_DIR}/*.jsonl", recursive=True):
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file):
            data = json.loads(line)
            file_name = os.path.basename(jsonl_file)
            if file_name not in datasets:
                datasets[file_name] = []
            datasets[file_name].append(data)


def clean_text(text: str) -> str:
    text = re.sub(r'\bAssistant\s*:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bAssistant\s*', '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'\bUser\s*:.*', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()


def build_prompt_from_data(data, turn):
    outputs = data.get("outputs", []) or []
    base_input = data.get("input", "") or ""
    instructions = data.get("instructions", []) or []
    msgs2 = [{'role': 'user', 'content': base_input + '\n\n'}]
    for i, question in enumerate(instructions):
        if i > turn:
            break
        if i == 0:
            msgs2[0]['content'] += question + "\n\nPlease think step by step."
        else:
            assistant_reply = outputs[i-1] if (i-1) < len(outputs) else ""
            msgs2.append({'role': 'assistant', 'content': assistant_reply})
            msgs2.append({'role': 'user', 'content': question + "\n\nPlease think step by step."})
    
    prompt = tokenizer.apply_chat_template(
        msgs2,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"rows: {len(df)}")

    rows = []
    for _, row in df.iterrows():
        file_name = row['file']
        line_number = int(row['line'])
        turn_allowed = int(row['turn'])

        data = datasets[file_name][line_number]
        prompt = build_prompt_from_data(data, turn_allowed)
        rows.append({
            "file": file_name,
            "line": line_number,
            "turn": turn_allowed,
            "prompt": prompt[:130700]
        })

    llm = LLM(model=MODEL_NAME, tensor_parallel_size=TENSOR_PARALLEL_SIZE)
    with open(OUTPUT_JSONL, "a", encoding="utf-8") as fout:
        for r in rows:
            outputs = llm.generate([r["prompt"]], SAMPLING_PARAMS)
            out = outputs[0]
            gen_text = out.outputs[0].text if len(out.outputs) > 0 else ""
            token_id = tokenizer.encode(gen_text, add_special_tokens=False)
            token_length = len(token_id)
            item = {
                "file": r["file"],
                "line": r["line"],
                "turn": r["turn"],
                "generate_text": gen_text,
                "tokens_length": token_length,
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()