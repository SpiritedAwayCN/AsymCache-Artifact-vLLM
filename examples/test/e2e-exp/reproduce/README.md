# AsymCache vLLM Experiments

This directory contains the LongBench and LooGLE replay interfaces, workload configurations, and result analyzers used by AsymCache. The unified vLLM source also contains the TraceA real-workload replay, BFCL agent workload, and hyper-parameter experiments.

The [AsymCache Artifact reproduction guide](https://github.com/PKU-DAIR/AsymCache-Artifact/blob/main/README.md) provides the source builds, packaged datasets, experiment launchers, and paper figure commands. The artifact checks out this repository at `vllm/`.

## Set up the artifact

```bash
git clone --recurse-submodules https://github.com/PKU-DAIR/AsymCache-Artifact.git
cd AsymCache-Artifact
export ARTIFACT_ROOT="$PWD"
export MODEL_8B_PATH=/path/to/Llama-3.1-8B-Instruct
export MODEL_70B_PATH=/path/to/Llama-3.1-70B-Instruct

conda create -n asymcache-vllm python=3.10 -y
conda activate asymcache-vllm
source scripts/env.sh
bash scripts/build.sh vllm
python scripts/prepare_data.py
```

The build uses CUDA Toolkit 12.8 and PyTorch 2.7.0. It compiles the customized FlashAttention source selected by `flash_attn_src` and `VLLM_FLASH_ATTN_SRC_DIR`, installs POD attention from `pod_attn_dev/pod_attn/`, and compiles `vllm/csrc/c_set/c_set.cpp`.

## Run the experiments

Run these commands from the artifact root in the `asymcache-vllm` environment:

| Experiment | Command |
| --- | --- |
| LongBench and LooGLE end-to-end results | `bash scripts/run.sh e2e` |
| Qwen-Bailian real workload, named TraceA in the code | `bash scripts/run.sh tracea` |
| BFCL agent workload and Continuum integration | `bash scripts/run.sh agent` |
| Hyper-parameter sensitivity | `bash scripts/run.sh sensitivity` |
| MSA kernel comparison | `python scripts/benchmark_kernel.py` |

Select a LongBench configuration with:

```bash
bash scripts/run.sh e2e --baseline ASYMCACHE --workload longbench \
  --model-size 8B --qps 0.02 --frac 5 --gpu 0 --port 8050
```

The shared environment points the replay interfaces to `data/workloads/`. LongBench uses `Longbench/` and `data_preprocess/longbench_deepseek_results.jsonl`; LooGLE uses `LooGLE_merged/` and `data_preprocess/loogle_deepseek_results.jsonl`. Arrival schedules are in `longbench_data/` and `loogle_data/`.

Replay records and logs are written under `results/raw/vllm/`. Run the SGLang baseline in the `asymcache-sglang` environment using the build and run commands in the artifact guide. Summarize the collected results and generate the paper figures with:

```bash
python plots/collect_results.py
python plots/plot_all.py
```

Summary tables are written to `results/summary/`, and PDF/PNG figures are written to `figures/`.
