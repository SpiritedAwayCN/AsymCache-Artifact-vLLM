import json
rows, total, missing= 0, 0, 0


def calculate_avg_tokens(file_path):
    prompt_tokens_list = []
    completion_tokens_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            prompt_tokens_list.append(data['prompt_tokens'])
            completion_tokens_list.append(data['completion_tokens'])
    avg_prompt_tokens = sum(prompt_tokens_list) / len(prompt_tokens_list)
    avg_completion_tokens = sum(completion_tokens_list) / len(completion_tokens_list)
    print("-" * 30)
    print(f"总样本数: {len(prompt_tokens_list)}")
    print(f"Prompt Tokens 平均值: {avg_prompt_tokens:.2f}")
    print(f"Completion Tokens 总和: {sum(completion_tokens_list)}")
    print(f"Completion Tokens 平均值: {avg_completion_tokens:.2f}")
    print(f"Prompt Tokens 最大值: {max(prompt_tokens_list)} | 最小值: {min(prompt_tokens_list)}")
    print(f"Completion Tokens 最大值: {max(completion_tokens_list)} | 最小值: {min(completion_tokens_list)}")
    print("-" * 30)


with open("./MWF_qps3.0/MWF_replay_output_qps3.0.jsonl", "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        usage = data.get("usage") or {}
        value = usage.get("completion_tokens")
        if value is None:
            missing += 1
            continue
        total += int(value)
        rows += 1

calculate_avg_tokens("../data/API_web_search_trace.jsonl")
print(f"rows_with_completion_tokens: {rows}")
print(f"missing_completion_tokens: {missing}")
print(f"completion_tokens_sum: {total}")