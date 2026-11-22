import json
import copy
import argparse
from collections import OrderedDict


def expand_results_jsonl(in_path: str, out_path: str, factor: int):
    records = []
    with open(in_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            records.append(json.loads(ln))

    by_file = OrderedDict()
    for rec in records:
        fname = rec.get("file")
        by_file.setdefault(fname, []).append(rec)

    out_lines = []
    for fname, recs in by_file.items():
        unique_lines_sorted = sorted({r["line"] for r in recs})
        base_index = {ln: idx for idx, ln in enumerate(unique_lines_sorted)}
        L = len(unique_lines_sorted)

        for r in recs:
            out_lines.append(json.dumps(r, ensure_ascii=False))

        for m in range(1, factor):
            for r in recs:
                nr = copy.deepcopy(r)
                old_line = nr["line"]
                idx = base_index[old_line]
                nr["line"] = idx + m * L
                out_lines.append(json.dumps(nr, ensure_ascii=False))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", default="./longbench_deepseek_results.jsonl")
    parser.add_argument("--out_path", default="./two_longbench_deepseek_results.jsonl")
    parser.add_argument("--factor", default=2)
    args = parser.parse_args()
    expand_results_jsonl(args.in_path, args.out_path, args.factor)
