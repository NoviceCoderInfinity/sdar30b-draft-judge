| cell | n | acc % | passes/blk | exp/token | +drafter | mean gen len | hit limit % | loop % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stock 1K | 90 | 6.7 | 10.77 | 998 | 998 | 830 | 35.6 | 15.6 |
| judge 1K | 90 | 5.6 | 8.33 | 670 | 776 | 786 | 31.1 | 13.3 |
| stock 4K | 90 | 7.8 | 8.50 | 781 | 781 | 1202 | 10.0 | 16.7 |
| judge 4K | 90 | 6.7 | 6.25 | 513 | 618 | 1180 | 10.0 | 15.6 |

paired 4K judge vs stock: n=90 Δ -1.1 [-5.6, +2.2]; speed-up passes 1.36x, judge-only 1.52x, judge+drafter 1.26x
stock right at 4K but wrong at 1K: 1
