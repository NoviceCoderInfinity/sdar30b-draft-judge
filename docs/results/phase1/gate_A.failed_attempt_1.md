# Checkpoint A gate

## Held-out probe, 32M -> 128M

| domain | conf. precision | conf. share | override conf. precision | improving? |
|---|---|---|---|---|
| maths | missing | | | |
| code | missing | | | |
| science | missing | | | |
| if | missing | | | |
| chat | missing | | | |

GSM8K-500 judge+DES48 at checkpoint A: missing on n=0 (released 90.2; must be >= 88.7) -> NOT OK

**Verdict: STOP (final evaluation on the 128M judge).** Improving domains: none.

## Checkpoint-A benchmarks

| cell | n | acc % | vs released | passes/blk | x fewer passes |
|---|---:|---:|---:|---:|---:|

## 32M test: drafter confidence as a judge input

| set | tau | main 32M acc / passes | conf 32M acc / passes |
|---|---|---|---|

Not run (the 32M drafter-confidence test was cut on the 17 h rental; see the extra-work notes).

## Evaluation jobs: finished, cut, skipped

| queue | cell | shards done | items | status |
|---|---|---:|---:|---|
| — | — | | | no evaluation queue was written (evaluations never started) |

## Learning-rate test at 8M tokens (held-out probe all/ce, lower is better)

Not run (no time before the main run: LR 8e-5 used).
