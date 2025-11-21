import json
from transformers import AutoTokenizer

def calculate_average_tokens_length(jsonl_file):
    count = 0
    total_tokens = 0
    tokens_list = []
    tokenizer = AutoTokenizer.from_pretrained("LargeWorldModel/LWM-Text-1M")
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                gen_text = data.get('generate_text')
                token_id = tokenizer.encode(gen_text, add_special_tokens=False)
                count += 1
                tokens_length = len(token_id)
                total_tokens += tokens_length
                tokens_list.append(tokens_length)
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
                continue
    
    if count > 0:
        average_tokens = total_tokens / count
        max_tokens = max(tokens_list)
        min_tokens = min(tokens_list) 
        print(f"统计结果:")
        print(f"总记录数: {count}")
        print(f"平均token长度: {average_tokens:.2f}")
        print(f"最大token长度: {max_tokens}")
        print(f"最小token长度: {min_tokens}")
        print(f"总token数: {total_tokens}")
    else:
        print("没有找到有效数据")

calculate_average_tokens_length("/jizhicfs/marvinhjia/sca/data_preprocess/longbench_deepseek_results.jsonl")