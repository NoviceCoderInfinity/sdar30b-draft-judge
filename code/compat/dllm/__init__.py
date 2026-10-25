"""SHIM -- not the real dLLM library (github.com/ZHZisZZ/dllm).

The dllm-hub checkpoints used as Phase 2 drafters (S1 Qwen3-0.6B-bd3lm, S2 Qwen2.5-Coder-0.5B-bd3lm) ship a
`modeling_qwen*.py` whose `if __name__ == "__main__":` demo does `import dllm`. transformers scans every import of a
trust_remote_code file and refuses to load it when one is missing, even though that block never runs. Nothing from
dllm is used at load or at inference; this empty package only satisfies the scan (code/compat is on sys.path in
every entry point).
"""
