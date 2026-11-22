# import json
# import glob
# import numpy as np
# import matplotlib.pyplot as plt
# from transformers import AutoTokenizer

# token_lengths = []
# model_name = "meta-llama/Llama-3.1-70B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=500000)
# files = glob.glob("/jizhicfs/marvinhjia/sca/LooGLE/*.jsonl")

# for file in files:
#     with open(file, "r", encoding="utf-8") as f:
#         for line in f:
#             data = json.loads(line)
#             text = data["context"] + " " + data.get("question", "")
#             tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
#             token_lengths.append(len(tokens))

# token_lengths = np.array(token_lengths)
# min_len = token_lengths.min()
# max_len = token_lengths.max()
# mean_len = token_lengths.mean()

# sorted_lengths = np.sort(token_lengths)
# cdf = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)

# plt.figure(figsize=(8, 6))
# plt.plot(sorted_lengths, cdf * 100, label="CDF")
# plt.xlabel("Token Length")
# plt.ylabel("Cumulative Percentage (%)")
# plt.title("Llama-3.1-70B(CDF)")

# plt.text(
#     0.95, 0.05,
#     f"Min: {min_len}\nMax: {max_len}\nMean: {mean_len:.2f}",
#     transform=plt.gca().transAxes,
#     ha="right", va="bottom",
#     fontsize=10,
#     bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
# )

# plt.grid(True, linestyle="--", alpha=0.7)
# plt.legend()
# plt.savefig("token_length_cdf.png", dpi=300, bbox_inches="tight")
# plt.close()
# print(f"统计完成！共 {len(token_lengths)} 条数据，图已保存为 token_length_cdf.png")
from datasets import load_dataset
import json
from collections import defaultdict

# 定义所有数据集
datasets = ["shortdep_qa", "shortdep_cloze", "longdep_qa", "summarization"]

# 初始化统计变量
total_stats = {
    'total_docs': 0,
    'total_qa_pairs': 0,
    'average_rounds': 0
}

dataset_stats = {}

print("开始统计每个doc_id的平均QA轮数:")
print("="*50)

for testset in datasets:
    print(f"\n正在处理数据集: {testset}")

    try:
        # 加载数据
        data = load_dataset('bigainlco/LooGLE', testset, split='test')

        # 按doc_id分组统计QA对数量
        doc_qa_count = defaultdict(int)

        for instance in data:
            doc_id = instance['doc_id']
            doc_qa_count[doc_id] += 1

        # 计算该数据集的统计信息
        total_docs = len(doc_qa_count)
        total_qa_pairs = sum(doc_qa_count.values())
        avg_rounds = total_qa_pairs / total_docs if total_docs > 0 else 0

        # 保存统计结果
        dataset_stats[testset] = {
            'total_documents': total_docs,
            'total_qa_pairs': total_qa_pairs,
            'average_rounds': avg_rounds,
            'doc_qa_distribution': dict(doc_qa_count)
        }

        print(f"  统计结果:")
        print(f"    文档数量: {total_docs}")
        print(f"    QA对总数: {total_qa_pairs}")
        print(f"    平均轮数: {avg_rounds:.2f}")

    except Exception as e:
        print(f"❌ 处理数据集 {testset} 时出错: {e}")
        continue

# 计算整体统计
if dataset_stats:
    all_docs = sum(stats['total_documents'] for stats in dataset_stats.values())
    all_qa_pairs = sum(stats['total_qa_pairs'] for stats in dataset_stats.values())
    overall_avg_rounds = all_qa_pairs / all_docs if all_docs > 0 else 0

    total_stats.update({
        'total_docs': all_docs,
        'total_qa_pairs': all_qa_pairs,
        'average_rounds': overall_avg_rounds
    })

    print(f"\n所有数据集汇总:")
    print(f"  总文档数: {total_stats['total_docs']}")
    print(f"  总QA对数: {total_stats['total_qa_pairs']}")
    print(f"  整体平均轮数: {total_stats['average_rounds']:.2f}")
