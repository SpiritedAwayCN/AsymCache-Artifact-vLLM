import json
import random
import tiktoken
from tqdm import tqdm

OUTPUT_PATH = "/jizhicfs/marvinhjia/sca/sharegpt_500.jsonl"
SOURCE_PATH = "/jizhicfs/marvinhjia/sca/ShareGPT_V3_unfiltered_cleaned_split.json"
TARGET_NUM = 500
MAX_CONTEXT = 16384
enc = tiktoken.get_encoding("cl100k_base")


def token_len(text):
    return len(enc.encode(text, disallowed_special=()))


def process_sample(sample):
    convs = sample.get("conversations", [])
    if len(convs) < 3:
        return None
    
    i = 0
    outputs = []
    instructions = []   
    while i < len(convs) - 1:
        if convs[i]["from"] == "human" and convs[i + 1]["from"] == "gpt":
            q = convs[i]["value"].strip()
            a = convs[i + 1]["value"].strip()
            if token_len(q) < 4 or token_len(a) < 4:
                i += 2
                continue
            instructions.append(q)
            outputs.append(a)
            i += 2
        else:
            i += 1

    if len(instructions) == 0:
        return None

    total_tokens = sum(token_len(x) for x in instructions) + sum(token_len(x) for x in outputs)

    if total_tokens > MAX_CONTEXT:
        return None

    return {
        "id": sample.get("id", ""),
        "instructions": instructions,
        "outputs": outputs
    }


def main():
    print("Loading dataset...")
    with open(SOURCE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed = []
    for sample in tqdm(data):
        item = process_sample(sample)
        if item:
            processed.append(item)

    print("Filtered samples:", len(processed))
    selected = random.sample(processed, min(TARGET_NUM, len(processed)))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Saved", len(selected), "samples to", OUTPUT_PATH)


if __name__ == "__main__":
    main()