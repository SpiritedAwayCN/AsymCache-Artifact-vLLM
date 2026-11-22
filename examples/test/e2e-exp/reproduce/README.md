# Reproduce Experiments

## 1. Description

Environment dependencies:

```text
CUDA: 12.8
CPU: x86_64
GPU: 8 x H20
PyTorch: 2.7.0
```

## 2. Installation

To clone this repository together with the vLLM submodule, use:

```bash
git clone --recursive https://gitee.com/cyllyl/multi-segment-attention.git
cd multi-segment-attention
```

If the repository has already been cloned without submodules, initialize the submodule manually:

```bash
git submodule update --init --recursive
```

After this step, the vllm/ directory should not be empty.

```bash
conda create -n msa python=3.10
conda activate msa
```

Install `pod_attn`:

```bash
cd pod_attn_dev
pip install -e .
```

Install the local vLLM package:

```bash
cd ../vllm
pip install -e . -v -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Build the `c_set` extension:

```bash
cd ./csrc/c_set
g++ -O3 -Wall -shared -std=c++17 -fPIC \
  $(python3 -m pybind11 --includes) \
  c_set.cpp \
  -o c_set$(python3-config --extension-suffix)
```

## 3. Run Experiments

Run the vLLM workloads from the `asymcache-unified` branch. Install the
checkout above and rebuild `c_set` from this branch to include the HOTPREFIX
extension. Set the model paths and Python executable for your machine.

```bash
export VLLM_ROOT=/path/to/vllm
export PYTHON_BIN=/path/to/conda/env/bin/python
export MODEL_8B_PATH=/path/to/Llama-3.1-8B-Instruct
export MODEL_70B_PATH=/path/to/Llama-3.1-70B-Instruct
export CSET_DIR="${VLLM_ROOT}/csrc/c_set"
export RESULTS_ROOT=/path/to/results
```

### (1) LongBench and LooGLE

Select an experiment with `--baseline` or `BASELINE`. The defaults run all
five groups across LongBench and LooGLE and both model sizes. Each group uses
the original workload configurations. `BS` is also accepted as `MAX_SCORE`.

```bash
cd "${VLLM_ROOT}/examples/test/e2e-exp/reproduce"
export VLLM_EXPERIMENT_DATA_ROOT=/path/to/prepared/reproduce/datasets
bash run_e2e.sh --baseline BS --workload longbench --model-size 8B --sweep 1
BASELINE=PENSIEVE bash run_e2e.sh --workload loogle --model-size 70B
bash run_e2e.sh --dry-run
```

`VLLM_EXPERIMENT_DATA_ROOT` defaults to `reproduce/datasets`. Supply the
prepared workload CSV files in `longbench_data/` and `loogle_data/`, the
raw inputs in `Longbench/` and `LooGLE_merged/`, and generated reference
outputs in `data_preprocess/`. The original configs target eight GPUs.
For a single sweep use `GPU` and `PORT` to match your machine.

The launcher maps baselines to the V0 engine settings below.

| Baseline | `VLLM_EVICTOR_POLICY` | `USE_MINEXP` | `VLLM_SIMPLE_SHED_ALGO` |
| --- | --- | --- | --- |
| LRU | `lru` | 0 | 0 |
| PENSIEVE | `pensieve` | 0 | 0 |
| BS | `bs` | 2 | 0 |
| ASYMCACHE | `mwf` | 2 | 1 |
| ASYMCACHE_NO_SCHEDULER | `mwf` | 2 | 0 |

Results are written to `${RESULTS_ROOT}/e2e/<group>/`. Generate the summary
from the shared results directory.

```bash
python result_analyse_all.py --results_dir "${RESULTS_ROOT}" \
  --output_csv "${RESULTS_ROOT}/summary.csv"
```

### (2) BFCL workload

Run the BFCL launcher from its own directory.

```bash
cd "${VLLM_ROOT}/examples/test/e2e-exp/BFCL_workload_reproduce"
```

Before running experiments on a new machine, edit `env.sh` or override its variables from the shell. The important variables are:

```text
PYTHON_BIN            # Python executable in the vLLM environment
MODEL_8B_PATH         # local path to Llama-3.1-8B-Instruct
MODEL_70B_PATH        # local path to Llama-3.1-70B-Instruct
HF_HOME               # Hugging Face cache root
HUGGINGFACE_HUB_CACHE # Hugging Face hub cache
CSET_DIR              # path required by the custom vLLM/pod_attention build
```

Data files:

```text
multi-segment-attention/BFCL_dataset/2x_bfcl_web_search_prompts_qps1.5.csv
multi-segment-attention/BFCL_dataset/2x_bfcl_web_search_prompts_qps1.75.csv
multi-segment-attention/BFCL_dataset/2x_bfcl_web_search_prompts_qps2.0.csv
```

Also, please modify the `DATA_DIR` variable in `run_bfcl_baseline_server.sh` according to your local repository path.

```bash
DATA_DIR=${DATA_DIR:-/path/to/multi-segment-attention/BFCL_dataset}
```

Use the shared launcher:

```bash
bash run_bfcl_baseline_server.sh --baseline MWF --model-size 70B --qps 1.75
```

The required command-line arguments are:

```text
--baseline    LRU, MWF, CONTINUUM, or CONTINUUM_MWF
--model-size  8B or 70B
--qps         1.5, 1.75, or 2.0
```

`BFCL_24_EXPERIMENT_COMMANDS.sh` contains the complete list of 24 commands. It is a copy-paste command list, not a script to run end-to-end. Do not run: `bash BFCL_24_EXPERIMENT_COMMANDS.sh`. Run one command at a time, or submit each command as a separate cluster job.

By default, outputs are written under:

```text
./${MODEL_SIZE}_results/2x_${BASELINE}_qps${QPS}/
```

For example:

```text
70B_results/2x_MWF_qps1.75/
|-- 2x_MWF_replay_output_qps1.75.jsonl
|-- schedule_qps1.75.log
|-- server_output_qps1.75.log
```

After the replay JSONL files are produced, summarize all 24 result directories:

```bash
python full_result.py
```

The summary CSV is written to:

```text
final_bfcl_24_summary.csv
```

### (3) TraceA

Place `mapping_config.json` next to the TraceA workload JSONL. Compute the
MINEXP parameters for each QPS with `CDF.py`. Pass the reported B, alpha and
beta to the launcher.

```bash
cd "${VLLM_ROOT}/examples/test/e2e-exp/Qwen_trace"
TRACEA_INPUT=/path/to/tracea_2k_cache_pressure_relaxed_2100.jsonl
"$PYTHON_BIN" CDF.py "$TRACEA_INPUT" --target-qps 0.2 --percentile 75

bash auto_script/run_traceA_2100.sh --baseline HOTPREFIX --model-size 8B \
  --qps 0.2 --aging-interval 2 --input-file "$TRACEA_INPUT"

bash auto_script/run_traceA_2100.sh --all-policies --model-size 70B \
  --qps 0.2 --input-file "$TRACEA_INPUT" \
  --minexp-b "$MINEXP_B" --minexp-alpha "$MINEXP_ALPHA" --minexp-beta "$MINEXP_BETA"
```

`--all-policies` runs LRU, PENSIEVE, BS, MWF and HOTPREFIX. Individual runs
also accept CONTINUUM and CONTINUUM_MWF. `--dry-run` prints the selected
engine settings and paths. Repeat with the parameters for QPS 0.25 and 0.3.

## 4. Hyper-parameter Sensitivity

Open the hyper-parameter sweep directory.

```bash
cd "${VLLM_ROOT}/examples/test/e2e-exp/hyper_parameter"
```

Each line in `ablation_config.txt` follows this format:

```text
<qps> <set> <grad> <minexp_b> <minexp_alpha> <minexp_beta> <dataset> <frac>
```

For example:

```text
0.02 0.6x 40 -6.456619e+02 6.257895e+04 1.564474e+03 longbench 5
```

Set any machine-specific overrides in your shell.

```bash
export CSET_DIR=/path/to/vllm/csrc/c_set
export BASE_DIR="${RESULTS_ROOT}/ablation_results"
export GPU=0
export PORT=8061
```

Please update these paths according to your local environment. Then, run:

```bash
bash run_ab_sweep.sh
```

For each configuration, `run_ab_one.sh` creates one output folder under:

```text
ablation_results/
```

The folder naming format is:

```text
LongBench_300_overall_qps<qps>_frac<frac>_set<set>_grad<grad>_gamma
```

For example:

```text
LongBench_300_overall_qps0.02_frac5_set0.6x_grad40_gamma
LongBench_300_overall_qps0.02_frac10_set1x1y_grad50_gamma
```

The provided configuration file contains 20 experiment settings, so the final result should contain 20 folders.

Finally, run:

```bash
python ../reproduce/result_analyse.py --folder "${BASE_DIR}" --output_csv ablation.csv
```

## 5. Plot Figures
