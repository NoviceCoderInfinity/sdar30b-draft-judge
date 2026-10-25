# Vendored scoring code

Source: https://github.com/JetAstra/SDAR @ main
Path:   evaluation/opencompass/opencompass/datasets/{math,gsm8k,mbpp}.py
Fetched: 2026-09-02

`oc_datasets/{math,gsm8k,mbpp}.py` are **unmodified copies**. `opencompass/` is a
shim supplying only: BaseEvaluator, ICL_EVALUATORS, LOAD_DATASET,
TEXT_POSTPROCESSORS, get_data_path, BaseDataset.

Rationale: P1 is a fidelity gate. Reimplementing `math_postprocess_v2` /
`MATHEvaluator(version='v2')` / `MBPPEvaluator` risks a scoring delta being
misread as a harness bug. Checksums in VENDORED.sha256.
