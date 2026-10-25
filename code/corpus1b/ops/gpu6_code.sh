#!/usr/bin/env bash
cd /workspace/dmge; set -a; . ./.env; set +a
export HF_HOME=/workspace/dmge/hf_home TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=6 env CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH SGLANG_FLASHINFER_WORKSPACE_SIZE=2147483648 \
  /venv/sgl/bin/python code/corpus1b/gen_sglang.py --wave A --domains code --stop-at 1790624852 --gpu 6 --num-gpus 8 \
  >> logs/c1b/gen_waveA_code.gpu6.log 2>&1
echo "gpu6 code gen exit $? $(date -u +%H:%M:%S)" >> logs/c1b/gen_waveA_code.gpu6.log
