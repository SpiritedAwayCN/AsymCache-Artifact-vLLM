import json

jsonl_path = "./correct_results/300_loogle_minexp_70B_overall_qps0.01_frac10_CV0.25_gamma/LooGLE_simulator_minexp_300_overall_qps0.01_frac10_CV0.25_P99_grad40_gamma_latter_70B.jsonl"
prompt_tokens_list = []
completion_tokens_list = []

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f):
        line = line.strip()
        if not line:
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[Warning] Line {line_idx} JSON decode failed: {e}")
            continue

        usage = item.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        if prompt_tokens is not None:
            prompt_tokens_list.append(prompt_tokens)
        if completion_tokens is not None:
            completion_tokens_list.append(completion_tokens)

def mean(lst):
    return sum(lst) / len(lst) if lst else 0

print(f"Total samples: {len(prompt_tokens_list)}")
print(f"Average prompt_tokens: {mean(prompt_tokens_list):.2f}")
print(f"Average completion_tokens: {mean(completion_tokens_list):.2f}")