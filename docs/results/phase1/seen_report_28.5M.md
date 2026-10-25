**Totals:** prompts on disk 2,239,372 (wave A 287,222); generated 287,222 answers (153.02M tokens incl. prompts); rejected 121,233 (77.91M); converted to blocks 165,438 items (74.76M); train split 164,438 items (74.36M); **seen 28.50M = 38% of converted train tokens**, items touched 65,867 (40%). Trainer count 28.50M.

### MATHS  — seen 7.70M of 21.57M converted train tokens (36%); items touched 15,552/43,668 (36%)

| group | source [subset] | prompts on disk | generated (wave A) | rejected | converted items (tokens) | seen tokens | % of converted seen | items touched |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **1 grade-school (GSM)** | **all** | **74,015** | **9,465** | **1,496** | **7,969 (2.14M)** | **0.77M** | **36%** | **36%** |
| | OpenMathInstruct-2 | 71,628 | 9,169 | 1,487 | 7,682 (2.07M) | 0.74M | 36% | 36% |
| | GSM8K train | 2,387 | 296 | 9 | 287 (0.07M) | 0.02M | 31% | 32% |
| **2 MATH-easy** | **all** | **262,853** | **33,730** | **18,804** | **14,690 (8.64M)** | **3.07M** | **36%** | **36%** |
| | OpenMathInstruct-2 | 262,000 | 33,620 | 18,798 | 14,592 (8.60M) | 3.05M | 36% | 36% |
| | MATH (Hendrycks) train Level 2 | 366 | 58 | 3 | 52 (0.02M) | 5.7K | 26% | 27% |
| | MATH (Hendrycks) train Level 3 | 343 | 37 | 2 | 32 (0.01M) | 6.2K | 43% | 42% |
| | MATH (Hendrycks) train Level 1 | 144 | 15 | 1 | 14 (4.4K) | 1.1K | 25% | 29% |
| **3 MATH-hard** | **all** | **281,476** | **36,122** | **18,286** | **17,544 (8.53M)** | **3.02M** | **36%** | **36%** |
| | NuminaMath-1.5 [cn_k12] | 103,515 | 13,332 | 7,638 | 5,692 (2.69M) | 0.96M | 36% | 36% |
| | DeepMath-103K | 74,754 | 9,558 | 5,328 | 4,014 (2.77M) | 0.96M | 35% | 35% |
| | NuminaMath-1.5 [orca_math] | 51,059 | 6,620 | 1,902 | 4,718 (1.53M) | 0.55M | 36% | 36% |
| | NuminaMath-1.5 [synthetic_math] | 47,761 | 6,080 | 2,998 | 3,010 (1.49M) | 0.53M | 36% | 35% |
| | NuminaMath-1.5 [metamath] | 3,665 | 453 | 383 | 70 (0.03M) | 8.9K | 29% | 33% |
| | MATH (Hendrycks) train Level 5 | 408 | 47 | 27 | 19 (0.01M) | 4.4K | 39% | 32% |
| | MATH (Hendrycks) train Level 4 | 314 | 32 | 10 | 21 (9.9K) | 4.1K | 41% | 38% |
| **4 olympiad** | **all** | **150,465** | **19,455** | **15,770** | **3,665 (2.36M)** | **0.84M** | **36%** | **36%** |
| | OpenMathReasoning | 64,000 | 8,331 | 7,398 | 923 (0.64M) | 0.22M | 35% | 34% |
| | NuminaMath-1.5 [olympiads] | 49,750 | 6,414 | 5,159 | 1,251 (0.75M) | 0.27M | 36% | 37% |
| | DeepMath-103K | 16,465 | 2,102 | 1,116 | 986 (0.64M) | 0.22M | 35% | 35% |
| | NuminaMath-1.5 [cn_contest] | 8,937 | 1,157 | 972 | 185 (0.12M) | 0.04M | 38% | 37% |
| | NuminaMath-1.5 [aops_forum] | 7,449 | 962 | 788 | 173 (0.12M) | 0.05M | 40% | 40% |
| | NuminaMath-1.5 [amc_aime] | 2,278 | 292 | 155 | 132 (0.09M) | 0.04M | 41% | 39% |
| | NuminaMath-1.5 [number_theory] | 588 | 77 | 72 | 5 (3.0K) | 1.5K | 49% | 60% |
| | NuminaMath-1.5 [inequalities] | 562 | 73 | 66 | 7 (5.8K) | 3.8K | 65% | 57% |
| | NuminaMath-1.5 [olympiads_ref] | 436 | 47 | 44 | 3 (2.1K) | 1.4K | 67% | 67% |

### CODE  — seen 10.08M of 33.86M converted train tokens (30%); items touched 19,765/66,357 (30%)

| group | source [subset] | prompts on disk | generated (wave A) | rejected | converted items (tokens) | seen tokens | % of converted seen | items touched |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **1 easier: function-level (Python)** | **all** | **696,382** | **89,234** | **23,011** | **66,223 (33.80M)** | **10.04M** | **30%** | **30%** |
| | OpenCodeInstruct [self-instruct, generic] | 292,896 | 37,623 | 7,547 | 30,076 (15.70M) | 4.68M | 30% | 30% |
| | OpenCodeInstruct [evol-instruct, generic] | 225,647 | 28,789 | 9,670 | 19,119 (10.00M) | 2.97M | 30% | 30% |
| | OpenCodeInstruct [self-instruct, algorithmic] | 151,596 | 19,450 | 4,360 | 15,090 (7.19M) | 2.13M | 30% | 30% |
| | OpenCodeInstruct [evol-instruct, algorithmic] | 25,945 | 3,342 | 1,418 | 1,924 (0.91M) | 0.26M | 28% | 28% |
| | MBPP train/val [train] | 233 | 24 | 12 | 12 (2.5K) | 1.4K | 57% | 42% |
| | MBPP train/val [validation] | 57 | 6 | 4 | 2 (0.5K) | 0 | 0% | 0% |
| **2 competitive (CF-style, stdin/stdout)** | **all** | **13,616** | **1,752** | **1,418** | **334 (0.16M)** | **0.05M** | **29%** | **29%** |
| | CodeContests [CODEFORCES] | 4,125 | 565 | 527 | 38 (0.02M) | 7.4K | 38% | 37% |
| | TACO-verified [EASY] | 1,929 | 247 | 143 | 104 (0.05M) | 0.01M | 23% | 24% |
| | CodeContests [HACKEREARTH] | 1,162 | 140 | 99 | 41 (0.02M) | 5.0K | 29% | 29% |
| | CodeContests [AIZU] | 1,088 | 147 | 130 | 17 (0.01M) | 3.6K | 35% | 29% |
| | CodeContests [ATCODER] | 971 | 130 | 100 | 30 (0.01M) | 2.2K | 21% | 20% |
| | TACO-verified [MEDIUM_HARD] | 771 | 102 | 89 | 13 (7.3K) | 4.0K | 55% | 54% |
| | CodeContests [CODECHEF] | 630 | 73 | 42 | 31 (0.01M) | 4.7K | 35% | 35% |
| | TACO-verified [UNKNOWN_DIFFICULTY] | 606 | 68 | 48 | 20 (8.9K) | 3.5K | 40% | 35% |
| | TACO-verified [HARD] | 602 | 72 | 66 | 6 (3.3K) | 0.9K | 28% | 17% |
| | TACO-verified [MEDIUM] | 593 | 87 | 68 | 19 (0.01M) | 1.3K | 10% | 11% |
| | TACO-verified [VERY_HARD] | 291 | 31 | 30 | 1 (0.4K) | 0 | 0% | 0% |
| | open-r1/codeforces [rating 800] | 101 | 13 | 9 | 4 (1.0K) | 0.2K | 20% | 25% |
| | APPS [introductory] | 84 | 12 | 4 | 8 (3.6K) | 1.2K | 32% | 38% |
| | open-r1/codeforces [rating 900] | 34 | 5 | 3 | 2 (0.9K) | 0.5K | 53% | 50% |

### SCIENCE  — seen 2.55M of 2.29M converted train tokens (112%); items touched 6,730/6,730 (100%)

| group | source [subset] | prompts on disk | generated (wave A) | rejected | converted items (tokens) | seen tokens | % of converted seen | items touched |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **free** | **all** | **213,112** | **27,344** | **23,061** | **4,283 (1.44M)** | **1.57M** | **112%** | **100%** |
| | OpenScienceReasoning-2 | 108,008 | 13,814 | 10,332 | 3,482 (1.05M) | 1.14M | 112% | 100% |
| | WebInstruct-verified [Physics] | 45,924 | 6,045 | 5,764 | 281 (0.14M) | 0.15M | 114% | 100% |
| | Nemotron-Science-v1 | 33,432 | 4,276 | 4,159 | 117 (0.08M) | 0.09M | 113% | 100% |
| | WebInstruct-verified [Chemistry] | 21,695 | 2,701 | 2,370 | 331 (0.15M) | 0.16M | 108% | 100% |
| | WebInstruct-verified [Biology] | 4,053 | 508 | 436 | 72 (0.01M) | 0.02M | 110% | 100% |
| **mc** | **all** | **54,130** | **6,998** | **4,351** | **2,647 (0.92M)** | **0.99M** | **112%** | **100%** |
| | OpenScienceReasoning-2 | 41,992 | 5,453 | 3,966 | 1,487 (0.47M) | 0.50M | 110% | 100% |
| | Nemotron-Science-v1 | 8,770 | 1,095 | 354 | 741 (0.32M) | 0.34M | 112% | 100% |
| | ai2_arc [ARC-Easy] | 2,250 | 293 | 19 | 274 (0.08M) | 0.10M | 117% | 100% |
| | ai2_arc [ARC-Challenge] | 1,118 | 157 | 12 | 145 (0.05M) | 0.05M | 113% | 100% |

### IF  — seen 2.64M of 2.73M converted train tokens (97%); items touched 8,362/8,652 (97%)

| group | source [subset] | prompts on disk | generated (wave A) | rejected | converted items (tokens) | seen tokens | % of converted seen | items touched |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **1 constraint(s)** | **all** | **83,237** | **10,669** | **4,421** | **6,248 (1.92M)** | **1.81M** | **97%** | **97%** |
| | WildChat-4.8M | 82,377 | 10,537 | 4,353 | 6,184 (1.90M) | 1.79M | 97% | 97% |
| | Nemotron-RL-instruction_following | 369 | 64 | 33 | 31 (0.01M) | 0.01M | 98% | 97% |
| | IF_multi_constraints_upto5 | 317 | 45 | 23 | 22 (7.5K) | 7.4K | 100% | 100% |
| | RLVR-IFeval | 174 | 23 | 12 | 11 (3.0K) | 3.0K | 100% | 100% |
| **2 constraint(s)** | **all** | **51,162** | **6,691** | **4,511** | **2,180 (0.73M)** | **0.69M** | **97%** | **97%** |
| | WildChat-4.8M | 49,044 | 6,410 | 4,301 | 2,109 (0.70M) | 0.66M | 97% | 97% |
| | Nemotron-RL-instruction_following | 1,960 | 259 | 192 | 67 (0.03M) | 0.03M | 98% | 97% |
| | IF_multi_constraints_upto5 | 158 | 22 | 18 | 4 (2.0K) | 2.0K | 100% | 100% |
| **3 constraint(s)** | **all** | **18,906** | **2,434** | **2,010** | **424 (0.15M)** | **0.14M** | **96%** | **96%** |
| | WildChat-4.8M | 18,000 | 2,331 | 1,918 | 413 (0.15M) | 0.14M | 96% | 96% |
| | Nemotron-RL-instruction_following | 824 | 97 | 87 | 10 (2.8K) | 2.3K | 100% | 100% |
| | IF_multi_constraints_upto5 | 82 | 6 | 5 | 1 (0.2K) | 0.2K | 100% | 100% |

### CHAT  — seen 5.52M of 13.91M converted train tokens (40%); items touched 15,458/39,031 (40%)

| group | source [subset] | prompts on disk | generated (wave A) | rejected | converted items (tokens) | seen tokens | % of converted seen | items touched |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **chat** | **all** | **340,018** | **43,328** | **4,094** | **39,231 (13.98M)** | **5.52M** | **40%** | **40%** |
| | WildChat-1M | 187,995 | 23,976 | 2,745 | 21,229 (6.26M) | 2.48M | 40% | 40% |
| | ultrachat_200k | 147,000 | 18,669 | 1,299 | 17,370 (7.57M) | 2.99M | 40% | 40% |
| | oasst2 | 5,023 | 683 | 50 | 632 (0.15M) | 0.05M | 33% | 36% |
