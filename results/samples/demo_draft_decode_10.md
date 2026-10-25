# Demo — the MoE denoising FROM the 1.7B draft, 9 GSM8K items

*2026-09-04 10:08 UTC. Block 32, threshold 0.95, bd3 semantics, drafter at 4 passes prefix-conditioned. Items = first 10 GSM8K prompts in `configs/prompts.json` (train split, reference answers). Normal decode = the blank-block trajectory already in `logs/P6`.*

**Rule (a) threshold-only:** on pass 0 the block holds the 32 draft tokens; commit the MoE's own prediction at every position with confidence > 0.95, re-mask the rest, then denoise normally. **Rule (b) agreement-only:** same, but commit only where the MoE's prediction *equals* the draft token. 'Draft words correct' = draft token equals the normal decode's token at that position, in the first full 32-token block.

> 1 item(s) not yet available: [515]

## Summary

| | rule (a) threshold | rule (b) agreement | normal decode |
|---|---|---|---|
| **correct, of 9** | **1** | **3** | **8** |
| mean passes per block | 12.07 | 15.01 | 17.11 |
| mean blocks per item | 10.4 | 8.0 | 7.0 |
| mean total passes per item | 126.1 | 120.1 | 119.8 |
| **items using FEWER total passes than normal** | **3/9** | **6/9** | — |
| passes ÷ normal passes (mean ratio) | 1.04 | 0.98 | 1.00 |
| **coherence: stopped by itself** | 5/9 | 8/9 | 9/9 |
| **coherence: produced a boxed answer** | 3/9 | 5/9 | 8/9 |
| **coherence: tail repetition (0 = none)** | 0.28 | 0.27 | 0.20 |
| block 1: draft words kept (of 32) | 0.2 | 0.1 | — |
| block 1: draft words correct (of 32) | 3.0 | 6.7 | — |
| block 1: kept AND correct | 0.1 | 0.1 | — |
| **kept ÷ correct** | **0.07** | **0.02** | — |

## Items

### GSM8K train #9  —  reference answer **990**

**1. Drafter, block 2:** 8.00 an hour. If she works 10 hours a day for 5 days, she will calculate her total earnings as overtime: /  0

**2a. MoE from draft, rule (a)** — ❌ wrong · 91 passes over 17 blocks [5, 16, 18, 12, 13, 8, 5, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1] (fewer than normal) · HIT LENGTH CAP · rep 0.22 · block 2: kept 1/32, correct 2/32, kept∧correct 1
> , Tina makes $1.00 an hour. If she works 10 hours every day for 5 days, we can calculate her total earnings as follows: / First, let's calculate her earnings for the first 8 hours each day: $18.000000000000000  * 8 = $18.00000000000000000000000000000000 for the 18.00000000000000000000000000000000 * 18.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 …

**2b. MoE from draft, rule (b)** — ❌ wrong · 89 passes over 17 blocks [5, 25, 11, 30, 2, 3, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1] (fewer than normal) · HIT LENGTH CAP · rep 0.97 · block 2: kept 1/32, correct 5/32, kept∧correct 1
> Tina makes $18.00 an hour and works 10 hours a day, so she earns 18 * 10 = $180 per day. / For the first 8 hours, she earns her hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hours hour …

**3. Normal blank-block decode** — ✅ correct · 108 passes over 6 blocks [4, 25, 21, 23, 19, 16] · stopped · rep 0.25
> Tina makes $18.00 an hour and works 10 hours a day, so she earns 18 * 10 = $180 per day. / For the first 8 hours, she earns 18 * 8 = $144. / For the remaining 2 hours (overtime), she earns 18 * 1.5 * 2 = $54. / So in total, she earns 144 + 54 = $198 per day. / If she works for 5 days, then she earns 198 * 5 = $990 in total. / The answer is $\boxed{990}$. /

### GSM8K train #115  —  reference answer **4000**

**1. Drafter, block 2:** , three fourths of the the jelly beans are red. So, the $\frac{{{3}{4}}}$x}$ the}$}$}$ jelly

**2a. MoE from draft, rule (a)** — ❌ wrong · 170 passes over 17 blocks [15, 23, 12, 16, 18, 20, 26, 23, 6, 2, 2, 2, 1, 1, 1, 1, 1] (MORE than normal) · HIT LENGTH CAP · rep 0.15 · block 2: kept 0/32, correct 6/32, kept∧correct 0
> Let's denote the total number of jelly beans in the jar as $x$. /  / According to the problem three fourths of the total jelly beans are red. So, the number of red jelly3/4 of the total jelly beans are red. So, the of red jelly beans is $\frac{3}{4}x$. /  / One quarter of the number of red jelly beans are coconut flavored. The coconut flavored jelly beans $\frac{1}{4} \times \frac{3}{4}x=\frac{3}{16}x$. /  / We know  …

**2b. MoE from draft, rule (b)** — ❌ wrong · 114 passes over 7 blocks [15, 18, 21, 17, 15, 15, 13] (fewer than normal) · stopped · rep 0.15 · block 2: kept 0/32, correct 6/32, kept∧correct 0
> Let's denote the total number of jelly beans in the jar as $x$. /  / According to the problem, three fourths of the jelly beans are red. So the number of red jelly beans is $\frac{3}{4}x$. /  / One quarter of the red jelly beans are coconut flavored. So the number of coconut flavored red jelly beans is $\frac{1}{4} \cdot \frac{3}{4}x = \frac{3}{16}x$. /  / We are given that there are 750 coconut flavored jelly beans, …

**3. Normal blank-block decode** — ✅ correct · 124 passes over 8 blocks [14, 17, 20, 16, 14, 14, 18, 11] · stopped · rep 0.12
> Let's denote the total number of jelly beans in the jar as $x$. /  / According to the problem, three fourths of the jelly beans are red. So the number of red jelly beans is $\frac{3}{4}x$. /  / One quarter of the red jelly beans are coconut flavored. So the number of coconut flavored red jelly beans is $\frac{1}{4} \cdot \frac{3}{4}x = \frac{3}{16}x$. /  / We are given that there are 750 coconut flavored jelly beans, …

### GSM8K train #132  —  reference answer **110**

**1. Drafter, block 2:** , are 4 pounds each. So each pack pack is 4 pounds. / The total price, the price of beef is $5.50 per pound

**2a. MoE from draft, rule (a)** — ❌ wrong · 38 passes over 3 blocks [3, 22, 13] (fewer than normal) · stopped · rep 0.15 · block 2: kept 0/32, correct 1/32, kept∧correct 0
> James buys5 packs of beef and 4 pounds of beef in each pack, so he bought a total of 5 * 4 = $5*$5.50 per pound, /  is /  /  /

**2b. MoE from draft, rule (b)** — ❌ wrong · 70 passes over 5 blocks [3, 20, 17, 20, 10] (fewer than normal) · stopped · rep 0.10 · block 2: kept 0/32, correct 9/32, kept∧correct 0
> James buys 5 packs of beef, and each pack is 4 pounds. So the total weight of the beef he bought is $5 \times 4 = 20$ pounds. / The price of beef is $5.50 per pound, so to find out how much he paid, we multiply the total weight by the price per pound: $20 \times \$5.50 = \$110$. / So James paid $\$110110$ dollars for the beef. /  namespace /

**3. Normal blank-block decode** — ✅ correct · 71 passes over 5 blocks [2, 19, 16, 16, 18] · stopped · rep 0.07
> James buys 5 packs of beef, and each pack is 4 pounds. So the total weight of the beef he bought is $5 \times 4 = 20$ pounds. / The price of beef is $5.50 per pound, so to find out how much he paid, we multiply the total weight by the price per pound: $20 \times 5.50 = 110$. / So James paid $\boxed{110}$ dollars for the beef. / import random / importimport time /

### GSM8K train #270  —  reference answer **600**

**1. Drafter, block 2:** are 60 minutes in in minute, the call call  60 minutes. If each each call costs five cents cents cents cents cents cents cents cents

**2a. MoE from draft, rule (a)** — ❌ wrong · 146 passes over 17 blocks [21, 21, 21, 17, 20, 20, 15, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1] (MORE than normal) · HIT LENGTH CAP · rep 0.18 · block 2: kept 0/32, correct 9/32, kept∧correct 0
> First, we need to find out how much each call costs. Since each call lasts 1 hour and there 60 minutes in an hour, each call lasts for 60 minutes. If each call costs 5 cents per minute, then the cost of each is 60 * 0.00000005, or 0.00000005 dollars. / Next we need to calculate how many calls Ayla makes in a month. If she talks 50 customers a week and there are 00000004 weeks in a month, then the total month's number …

**2b. MoE from draft, rule (b)** — ❌ wrong · 129 passes over 6 blocks [21, 22, 24, 13, 28, 21] (fewer than normal) · stopped · rep 0.18 · block 2: kept 0/32, correct 9/32, kept∧correct 0
> First, we need to find out how much each call costs. Since each call lasts 1 hour and there are 60 minutes in an hour, each call lasts 60 minutes. If each minute costs five cents, then each call costs 60 * $0.05 = $3. / Next, we need to find out how many customers Ayla talks to in a month. If she talks to 50 customers a week and there are 4 weeks in a month, then she talks to 50 * 4 = 200 customers in a month. / Fina …

**3. Normal blank-block decode** — ✅ correct · 136 passes over 7 blocks [20, 21, 23, 12, 21, 16, 23] · stopped · rep 0.30
> First, we need to find out how much each call costs. Since each call lasts 1 hour and there are 60 minutes in an hour, each call lasts 60 minutes. If each minute costs five cents, then each call costs 60 * $0.05 = $3. / Next, we need to find out how many customers Ayla talks to in a month. If she talks to 50 customers a week and there are 4 weeks in a month, then she talks to 50 * 4 = 200 customers in a month. / Fina …

### GSM8K train #316  —  reference answer **253**

**1. Drafter, block 2:** % increase. /  the original price price of $ bicycle200 /  the increase%% is 15% increase is 200 * 0

**2a. MoE from draft, rule (a)** — ❌ wrong · 49 passes over 3 blocks [3, 30, 16] (fewer than normal) · stopped · rep 0.21 · block 2: kept 1/32, correct 1/32, kept∧correct 0
> find the new price of the bicycle after a 15 increase from $220222220 + 220 *0.220 + 220 * 0.152220 + 33 = 253 / The new price is $\boxed /  /  /

**2b. MoE from draft, rule (b)** — ✅ correct · 99 passes over 7 blocks [10, 23, 18, 19, 8, 10, 11] (fewer than normal) · stopped · rep 0.30 · block 2: kept 0/32, correct 6/32, kept∧correct 0
> To find the new price of the bicycle after a 15% increase, we first need to calculate the amount of increase. The original price of the bicycle is $220, and the increase is 15% of this price. /  / The increase can be calculated as follows: / $$ \text{Increase} = \text{Original Price} \times \text{Percentage Increase} $$ / $$ = 220 \times 0.15 $$ / $$ = 33 $$ /  / Now, to find the new price, we add this increase to th …

**3. Normal blank-block decode** — ✅ correct · 112 passes over 8 blocks [9, 22, 17, 21, 11, 6, 8, 18] · stopped · rep 0.38
> To find the new price of the bicycle after a 15% increase, we first need to calculate the amount of increase. The original price of the bicycle is $220, and the increase is 15% of this price. /  / The increase can be calculated as follows: / $$ \text{Increase} = \text{Original Price} \times \text{Percentage Increase} $$ / $$ \text{Increase} = 220 \times 0.15 $$ /  / Let's calculate this: / $$ \text{Increase} = 220 \t …

### GSM8K train #331  —  reference answer **6**

**1. Drafter, block 2:** used for each day and the paving of the amount of the the third day. /  / First, let the the the the of the the the the the the the the

**2a. MoE from draft, rule (a)** — ❌ wrong · 229 passes over 17 blocks [18, 30, 23, 23, 29, 22, 31, 24, 19, 2, 2, 1, 1, 1, 1, 1, 1] (MORE than normal) · HIT LENGTH CAP · rep 0.97 · block 2: kept 0/32, correct 3/32, kept∧correct 0
> solve this problem, we need to break it down into smaller steps and calculate the amount of gravel and pitch needed for each day of work. /  / **Step 1: Calculation**. /  / First, let's calculate the total miles of road that the company paved on the first day: / - 4 miles /  / Next, let's calculate the total miles of road that the company paved on the second day: / - One mile less than double the miles of the first d …

**2b. MoE from draft, rule (b)** — ❌ wrong · 224 passes over 11 blocks [15, 19, 16, 17, 20, 27, 31, 24, 23, 19, 13] (MORE than normal) · stopped · rep 0.25 · block 2: kept 0/32, correct 9/32, kept∧correct 0
> The company paved 4 miles of road on the first day, so they used 4 * 3 = 12 truckloads of asphalt. / On the second day, they paved one mile less than double the amount of the first day, which is 2*4 - 1 = 7 miles. So they used 7 * 3 = 21 truckloads of asphalt on the second day. / In total, they used 12 + 21 = 33 truckloads of asphalt in the first two days. / The road is 16 miles long, and they have already paved 4 +  …

**3. Normal blank-block decode** — ❌ wrong · 186 passes over 10 blocks [14, 18, 15, 16, 19, 19, 26, 27, 14, 18] · stopped · rep 0.22
> The company paved 4 miles of road on the first day, so they used 4 * 3 = 12 truckloads of asphalt. / On the second day, they paved one mile less than double the amount of the first day, which is 2*4 - 1 = 7 miles. So they used 7 * 3 = 21 truckloads of asphalt on the second day. / In total, they used 12 + 21 = 33 truckloads of asphalt in the first two days. / The road is 16 miles long, and they have already paved 4 +  …

### GSM8K train #370  —  reference answer **64**

**1. Drafter, block 2:** 2 patients in her hospital ward Most of them standard care care patients require standard care standard care care patients, but one-third of her patients have special dietary requirements,

**2a. MoE from draft, rule (a)** — ❌ wrong · 193 passes over 9 blocks [4, 21, 22, 28, 29, 28, 29, 26, 6] (MORE than normal) · stopped · rep 0.20 · block 2: kept 0/32, correct 1/32, kept∧correct 0
> ,y is attending to the needs of 1 patients in her hospital ward. of these patients, most patients require standard care, but it is given that one-third of her patients have special dietary requirements, which the serving time by 20%. Therefore, the number of standard care patients is $12 - \frac{1}{3} \times 1 = 12 - 4 = Each standard care patient takes 5 minutes, it takes 5 minutes, so the total time taken to serve  …

**2b. MoE from draft, rule (b)** — ❌ wrong · 196 passes over 10 blocks [8, 22, 18, 21, 25, 26, 25, 20, 16, 15] (MORE than normal) · stopped · rep 0.22 · block 2: kept 0/32, correct 3/32, kept∧correct 0
> Nurse Missy is attending to the needs of 12 patients in her hospital ward. One-third of her patients have special dietary requirements, so the number of patients with special dietary requirements is $\frac{1}{3} \times 12 = 4$. / The remaining patients, which is $12 - 4 = 8$, require standard care. / It takes 5 minutes to serve each standard care patient, so it will take $8 * \times 5 = 40$ minutes to serve all the s …

**3. Normal blank-block decode** — ✅ correct · 181 passes over 10 blocks [7, 21, 17, 20, 24, 22, 17, 18, 17, 18] · stopped · rep 0.12
> Nurse Missy is attending to the needs of 12 patients in her hospital ward. One-third of her patients have special dietary requirements, so the number of patients with special dietary requirements is $\frac{1}{3} \times 12 = 4$. / The remaining patients, which is $12 - 4 = 8$, require standard care. / It takes 5 minutes to serve each standard care patient, so it will take $8 * \times 5 = 40$ minutes to serve all the s …

### GSM8K train #509  —  reference answer **6**

**1. Drafter, block 2:** , we need need to to first calculate the total consumption ofowatt-hours of the the the oven. the oven oven the consumption consumption is the oven the consumption

**2a. MoE from draft, rule (a)** — ❌ wrong · 135 passes over 7 blocks [6, 26, 19, 25, 26, 25, 8] (MORE than normal) · stopped · rep 0.15 · block 2: kept 0/32, correct 4/32, kept∧correct 0
> To find out how much Coco will pay for using his oven we need to multiply the cost of electricity price per kilowatt-hour by the total number of kilowatt-hours consumed. / The electricity price is $0.01 per kWh and Coco has a consumption rate of 2.4 kWh per hour. The total consumption for the 25 hours will be 24 * 25 = 600 kWh. / To find out how much Coco will pay, we multiply the total consumption (600 kWh) the elec …

**2b. MoE from draft, rule (b)** — ✅ correct · 104 passes over 6 blocks [6, 25, 21, 17, 24, 11] (MORE than normal) · stopped · rep 0.10 · block 2: kept 0/32, correct 4/32, kept∧correct 0
> To find out how much Coco will pay for using his oven, we first need to calculate the total energy consumed by the oven. The oven has a consumption rate of 2.4 kWh per hour and was used for 25 hours last month. So the total energy consumed is $2.4 \times 25 = 60$ kWh. / The electricity price in Coco's town is $0.10 per kWh. To find out how much Coco will pay, we multiply the total energy consumed by the price per kWh …

**3. Normal blank-block decode** — ✅ correct · 103 passes over 6 blocks [5, 24, 20, 16, 23, 15] · stopped · rep 0.07
> To find out how much Coco will pay for using his oven, we first need to calculate the total energy consumed by the oven. The oven has a consumption rate of 2.4 kWh per hour and was used for 25 hours last month. So the total energy consumed is $2.4 \times 25 = 60$ kWh. / The electricity price in Coco's town is $0.10 per kWh. To find out how much Coco will pay, we multiply the total energy consumed by the price per kWh …

### GSM8K train #510  —  reference answer **180**

**1. Drafter, block 2:** pie is 111 slices. So, the number of 111 slices is 6* 11 111111

**2a. MoE from draft, rule (a)** — ✅ correct · 84 passes over 4 blocks [29, 22, 19, 14] (MORE than normal) · stopped · rep 0.32 · block 2: kept 0/32, correct 0/32, kept∧correct 0
> The pie shop makes 6 whole pies, and each whole pie cut into 6 whole pies into 6 whole custard pies, and each whole pie is cut into 10 slices for a total of 6 x 10 = 60 slices. / Each slice of custard pie costs $3, so the pie shop will earn 60 slices slices x 60 x $3 per slice $3 = $180. / The pie will earn $180 dollars. /  is $\boxed{180}$. /  /

**2b. MoE from draft, rule (b)** — ✅ correct · 56 passes over 3 blocks [20, 19, 17] (fewer than normal) · stopped · rep 0.20 · block 2: kept 0/32, correct 9/32, kept∧correct 0
> If each whole pie is cut into 10 slices, and they make 6 whole pies, then the total number of slices is $6 \times 10 = 60$. / If each slice is sold for $3, then the total amount of money earned is $60 \times 3 = 180$. / The pie shop will earn $\boxed{180}$ dollars. /

**3. Normal blank-block decode** — ✅ correct · 57 passes over 3 blocks [19, 18, 20] · stopped · rep 0.22
> If each whole pie is cut into 10 slices, and they make 6 whole pies, then the total number of slices is $6 \times 10 = 60$. / If each slice is sold for $3, then the total amount of money earned is $60 \times 3 = 180$. / So, the pie shop will earn $\boxed{180}$ dollars. /

