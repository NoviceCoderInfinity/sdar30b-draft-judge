#!/bin/bash
# Build the round-2 training index = round-1 GSM8K set + MATH + MBPP + remaining GSM8K (all DV-blind q50 decodes,
# correct answers only, frozen prompts excluded). One build per source, then the index files are concatenated.
set -eu; cd "$(dirname "$0")/../.."
PY=/venv/main/bin/python; OUT=logs/draftalign/trainset_union_r2; mkdir -p $OUT; : > $OUT/index.jsonl
build(){ # $1 records dir, $2 split name, $3 out sub
  $PY code/draftalign/build_train_set.py --records "$1" --traces "$1/traces" --layout da --split-name "$2" --out "$OUT/$3" --only-correct --max-seq 1200 --exclude-frozen > "$OUT/$3.build.log" 2>&1
  $PY -c "import json,sys
for l in open('$OUT/$3/index.jsonl'):
    r=json.loads(l); r['file']='$3/'+r['file']; r['split']='$3:'+r['split']; print(json.dumps(r))" >> $OUT/index.jsonl; echo "$3: $($PY -c "import json;print(json.load(open('$OUT/$3/build_summary.json'))['summary'])")"
}
build logs/draftalign/datagen/gsm8k_pool_0_3000_dv-s8_blind_q50 "gsm8k_pool:0:3000" gsm8k_a
build logs/draftalign/datagen/gsm8k_pool_3000_7273_dv-s8_blind_q50 "gsm8k_pool:3000:7273" gsm8k_b
build logs/draftalign/datagen/math_pool_0_7500_3_dv-s8_blind_q50 "math_pool:0:7500:3" math
build logs/draftalign/datagen/mbpp_pool_dv-s8_blind_q50 "mbpp_pool" mbpp
echo "union index: $(wc -l < $OUT/index.jsonl) items -> $OUT/index.jsonl"
