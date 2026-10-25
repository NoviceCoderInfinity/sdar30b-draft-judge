#!/usr/bin/env bash
# parallel pre-fetch of every dataset (stage 1's own download step then finds them cached); one manifest part each
cd /workspace/dmge; set -a; . ./.env; set +a
export HF_HOME=/workspace/dmge/hf_home HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TOKENIZERS_PARALLELISM=false
G=(
 "gsm8k,gsm8k_test,mathtrain,math_test,math500,aime_22_24,aime_25,gpqa_diamond,gpqa_all,humaneval,mbpptrain,mbpp_all,ifeval,ifbench,finqa_test,finqa500,chartqa300,arc,webinst,ifnemo,ifmulti,ifrlvr"
 "cf"
 "omr"
 "omi2gsm,omi2math"
 "osr2"
 "ifcomp"
 "cc,oci"
 "numina,numinaolym,nemosci_rqa,nemosci_mcq,deepmath,taco,apps,wildchat,ultrachat,oasst2,chartqa_test"
)
pids=()
for i in "${!G[@]}"; do
  /venv/train/bin/python code/corpus1b/download.py --only "${G[$i]}" --include-optional \
     --out data/c1b/dl_parts/manifest_$i.json > logs/c1b/dl/dl_$i.log 2>&1 & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
echo "parallel download finished rc=$rc $(date -u)" | tee logs/c1b/dl/DONE
