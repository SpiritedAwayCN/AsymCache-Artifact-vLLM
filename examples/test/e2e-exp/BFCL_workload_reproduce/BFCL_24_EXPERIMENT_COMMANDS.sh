#!/bin/bash

# This file is a command list for reviewers.
# Do NOT run this whole script directly. Each command starts a vLLM server and
# blocks until manually stopped. Copy exactly one command at a time into a
# terminal or submit each line as a separate job.

echo "Do NOT execute this file directly. Copy one experiment command at a time."
exit 1

# 8B, qps=1.5
bash run_bfcl_baseline_server.sh --baseline LRU --model-size 8B --qps 1.5
bash run_bfcl_baseline_server.sh --baseline MWF --model-size 8B --qps 1.5
bash run_bfcl_baseline_server.sh --baseline CONTINUUM --model-size 8B --qps 1.5
bash run_bfcl_baseline_server.sh --baseline CONTINUUM_MWF --model-size 8B --qps 1.5

# 8B, qps=1.75
bash run_bfcl_baseline_server.sh --baseline LRU --model-size 8B --qps 1.75
bash run_bfcl_baseline_server.sh --baseline MWF --model-size 8B --qps 1.75
bash run_bfcl_baseline_server.sh --baseline CONTINUUM --model-size 8B --qps 1.75
bash run_bfcl_baseline_server.sh --baseline CONTINUUM_MWF --model-size 8B --qps 1.75

# 8B, qps=2.0
bash run_bfcl_baseline_server.sh --baseline LRU --model-size 8B --qps 2.0
bash run_bfcl_baseline_server.sh --baseline MWF --model-size 8B --qps 2.0
bash run_bfcl_baseline_server.sh --baseline CONTINUUM --model-size 8B --qps 2.0
bash run_bfcl_baseline_server.sh --baseline CONTINUUM_MWF --model-size 8B --qps 2.0

# 70B, qps=1.5
bash run_bfcl_baseline_server.sh --baseline LRU --model-size 70B --qps 1.5
bash run_bfcl_baseline_server.sh --baseline MWF --model-size 70B --qps 1.5
bash run_bfcl_baseline_server.sh --baseline CONTINUUM --model-size 70B --qps 1.5
bash run_bfcl_baseline_server.sh --baseline CONTINUUM_MWF --model-size 70B --qps 1.5

# 70B, qps=1.75
bash run_bfcl_baseline_server.sh --baseline LRU --model-size 70B --qps 1.75
bash run_bfcl_baseline_server.sh --baseline MWF --model-size 70B --qps 1.75
bash run_bfcl_baseline_server.sh --baseline CONTINUUM --model-size 70B --qps 1.75
bash run_bfcl_baseline_server.sh --baseline CONTINUUM_MWF --model-size 70B --qps 1.75

# 70B, qps=2.0
bash run_bfcl_baseline_server.sh --baseline LRU --model-size 70B --qps 2.0
bash run_bfcl_baseline_server.sh --baseline MWF --model-size 70B --qps 2.0
bash run_bfcl_baseline_server.sh --baseline CONTINUUM --model-size 70B --qps 2.0
bash run_bfcl_baseline_server.sh --baseline CONTINUUM_MWF --model-size 70B --qps 2.0
