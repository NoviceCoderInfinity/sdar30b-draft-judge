# Checkpoint A gate

Checkpoint A = `/workspace/dmge/logs/c1b/train/main/final` at 11.79M corpus tokens (probe tag `final`).

## Held-out probe, 32M -> 128M

| domain | conf. precision | conf. share | override conf. precision | improving? |
|---|---|---|---|---|
| maths | missing | | | |
| code | missing | | | |
| science | missing | | | |
| if | missing | | | |
| chat | missing | | | |

GSM8K-500 judge+DES48 at checkpoint A: 91.0 on n=500 (released 90.2; must be >= 88.7) -> OK

**Verdict: STOP (final evaluation on the 128M judge).** Improving domains: none.

> The pre-registered probe criterion (32M -> 128M) is not evaluable: training stopped at 11.79M tokens on this rental. Supplementary, same test on the probes that exist (tok_000001840000 @ 1.84M -> final @ 11.79M):

| domain | conf. precision | conf. share | override conf. precision | improving? |
|---|---|---|---|---|
| maths | 99.3 -> 99.3 | 30.3 -> 33.6 | 90.2 -> 88.6 | yes |
| code | 99.1 -> 99.3 | 46.2 -> 51.6 | 91.9 -> 93.5 | yes |
| science | 98.7 -> 98.8 | 22.7 -> 24.7 | 91.5 -> 91.1 | yes |
| if | 97.9 -> 98.2 | 9.1 -> 11.0 | 86.4 -> 89.5 | yes |
| chat | 98.6 -> 99.0 | 12.2 -> 13.1 | 82.9 -> 89.3 | yes |

Supplementary reading: improving in code, science, if of code / science / if; with the GSM8K criterion met, the gate would read **CONTINUE** on these probes (not the pre-registered one).

## Checkpoint-A benchmarks

| cell | n | acc % | vs released | passes/blk | x fewer passes |
|---|---:|---:|---:|---:|---:|
| aime_judge128M_des48 | 90 | 6.7 | -0.0 | 8.91 | 1.21 |
| chartqa300_judge128M_des48 | 300 | 44.0 | — | 10.71 | — |
| chartqa300_released | 300 | 56.0 | — | 12.55 | — |
| finqa500_judge128M_des48 | 376 | 27.1 | — | 10.45 | — |
| finqa500_released | 376 | 28.5 | — | 12.78 | — |
| gpqa_judge128M_des48 | 174 | 31.6 | -5.8 | 14.59 | 1.10 |
| gsm8k500_judge128M_des48 | 500 | 91.0 | 0.8 | 7.72 | 1.30 |
| humaneval_judge128M_des48 | 164 | 73.8 | -4.2 | 5.96 | 1.39 |
| ifeval541_judge128M_des48 | 407 | 38.3 | — | 16.48 | — |
| ifeval541_released | 407 | 51.1 | — | 20.47 | — |
| math500_judge128M_des48 | 438 | 72.8 | 0.4 | 6.95 | 1.23 |
| mbpp_judge128M_des48 | 257 | 68.5 | 3.1 | 9.58 | 1.25 |

## 32M test: drafter confidence as a judge input

| set | tau | main 32M acc / passes | conf 32M acc / passes |
|---|---|---|---|

Not run (the 32M drafter-confidence test was cut on the 17 h rental; see the extra-work notes).

## Evaluation jobs: finished, cut, skipped

| queue | cell | shards done | items | status |
|---|---|---:|---:|---|
| checkpointA | gsm8k500_judge128M_des48 | 8/8 | 500 | complete |
| checkpointA | humaneval_judge128M_des48 | 4/4 | 164 | complete |
| checkpointA | mbpp_judge128M_des48 | 4/4 | 257 | complete |
| checkpointA | math500_judge128M_des48 | 7/8 | 438 | partial: cut at DEADLINE or unfinished |
| checkpointA | aime_judge128M_des48 | 4/4 | 90 | complete |
| checkpointA | gpqa_judge128M_des48 | 7/8 | 174 | partial: cut at DEADLINE or unfinished |
| checkpointA | ifeval541_judge128M_des48 | 6/8 | 407 | partial: cut at DEADLINE or unfinished |
| checkpointA | finqa500_judge128M_des48 | 6/8 | 376 | partial: cut at DEADLINE or unfinished |
| checkpointA | chartqa300_judge128M_des48 | 4/4 | 300 | complete |
| checkpointA | ifeval541_released | 6/8 | 407 | partial: cut at DEADLINE or unfinished |
| checkpointA | finqa500_released | 6/8 | 376 | partial: cut at DEADLINE or unfinished |
| checkpointA | chartqa300_released | 4/4 | 300 | complete |

## Learning-rate test at 8M tokens (held-out probe all/ce, lower is better)

| LR | probe all/ce | source | tokens |
|---|---:|---|---:|
| 8e-5 | 2.5750 | main run @ 8M (same schedule and data order) | 8.00M |
