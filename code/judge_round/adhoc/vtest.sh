#!/bin/bash
cd /workspace/diffusion-moe-general-expert
export HF_HOME=/workspace/diffusion-moe-general-expert/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export C1B_CODE_DIR=/workspace/diffusion-moe-general-expert/code
. logs/phase5/sgl_setup/sgl_env.sh
export CUDA_VISIBLE_DEVICES=7 SGLANG_FLASHINFER_WORKSPACE_SIZE=4294967296
OUT=/workspace/diffusion-moe-general-expert/logs/phase5/judge_sgl/vanilla_sgl_gsm8k500
mkdir -p $OUT
echo "start $(TZ=Asia/Kolkata date +%H:%M:%S) IST" > $OUT/run_times.txt
/venv/sgl/bin/python code/corpus1b/gen_sglang.py --items gsm8k500 --gpu 0 --num-gpus 1 --out $OUT > $OUT/gen.log 2>&1
echo "exit=$? end $(TZ=Asia/Kolkata date +%H:%M:%S) IST" >> $OUT/run_times.txt
