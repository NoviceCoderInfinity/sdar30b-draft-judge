# Demo — the MoE denoising FROM the 1.7B draft, 20 GSM8K items

*2026-09-04 10:10 UTC. Block 32, threshold 0.95, bd3 semantics, drafter at 4 passes prefix-conditioned. Items = first 20 GSM8K prompts in `configs/prompts.json` (train split, reference answers). Normal decode = the blank-block trajectory already in `logs/P6`.*

**Rule (a) threshold-only:** on pass 0 the block holds the 32 draft tokens; commit the MoE's own prediction at every position with confidence > 0.95, re-mask the rest, then denoise normally. **Rule (b) agreement-only:** same, but commit only where the MoE's prediction *equals* the draft token. 'Draft words correct' = draft token equals the normal decode's token at that position, in the first full 32-token block.

## Summary

| | rule (a) threshold | rule (b) agreement | normal decode |
|---|---|---|---|
| **correct, of 20** | **2** | **5** | **15** |
| mean passes per block | 11.38 | 10.65 | 17.87 |
| mean blocks per item | 10.3 | 10.7 | 7.0 |
| mean total passes per item | 117.8 | 114.0 | 126.0 |
| **items using FEWER total passes than normal** | **11/20** | **15/20** | — |
| passes ÷ normal passes (mean ratio) | 0.94 | 0.90 | 1.00 |
| **coherence: stopped by itself** | 12/20 | 12/20 | 20/20 |
| **coherence: produced a boxed answer** | 8/20 | 8/20 | 19/20 |
| **coherence: tail repetition (0 = none)** | 0.33 | 0.28 | 0.23 |
| block 1: draft words kept (of 32) | 0.5 | 0.7 | — |
| block 1: draft words correct (of 32) | 3.6 | 6.8 | — |
| block 1: kept AND correct | 0.1 | 0.2 | — |
| **kept ÷ correct** | **0.12** | **0.10** | — |

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

### GSM8K train #515  —  reference answer **24**

**1. Drafter, block 2:** many degrees temperature temperature  22 degrees below below 22 degrees. 332 degrees. 32 degrees degrees 33

**2a. MoE from draft, rule (a)** — ❌ wrong · 144 passes over 17 blocks [9, 27, 20, 28, 28, 17, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1] (fewer than normal) · HIT LENGTH CAP · rep 0.10 · block 2: kept 0/32, correct 3/32, kept∧correct 0
> First, let's help Annie to figure out how many degrees the temperature is below 32 degrees degrees. The temperature is 8 degrees, so it's 32 - 8 = 24 below 32 degrees. / Next, since for every 3 degrees the temperature drops below 32 degrees, Annie's chances of skidding on ice increase5%, we can calculate 24 degrees divided by 3 (4 times) and multiply that by 5% to find out how much her chances ofidding555555555555555 …

**2b. MoE from draft, rule (b)** — ❌ wrong · 147 passes over 7 blocks [10, 25, 18, 27, 19, 23, 25] (fewer than normal) · stopped · rep 0.22 · block 2: kept 0/32, correct 12/32, kept∧correct 0
> First, we need to find out how many degrees the temperature is below 32 degrees. The temperature is 8 degrees, so it is $32 - 8 = 24$ degrees below 32 degrees. / For every 3 degrees the temperature drops below 32 degrees, Annie's chances of skidding on ice increase 5%. So for 24 degrees below 32 degrees, her chances of skidding increase by $24/3 \times 5\% = 40\%$. / If she goes into a skid, she has a 40% chance of r …

**3. Normal blank-block decode** — ✅ correct · 171 passes over 9 blocks [9, 24, 17, 26, 18, 23, 16, 15, 23] · stopped · rep 0.20
> First, we need to find out how many degrees the temperature is below 32 degrees. The temperature is 8 degrees, so it is $32 - 8 = 24$ degrees below 32 degrees. / For every 3 degrees the temperature drops below 32 degrees, Annie's chances of skidding on ice increase 5%. So for 24 degrees below 32 degrees, her chances of skidding increase by $24/3 \times 5\% = 40\%$. / If she goes into a skid, she has a 40% chance of r …

### GSM8K train #565  —  reference answer **294**

**1. Drafter, block 2:** 3 times the amount that Katie drives drives $3x$, $3xxx$ thriceice 3000ice00 00

**2a. MoE from draft, rule (a)** — ❌ wrong · 126 passes over 6 blocks [17, 22, 24, 24, 19, 20] (fewer than normal) · stopped · rep 0.30 · block 2: kept 0/32, correct 9/32, kept∧correct 0
> Let's denote the number of miles Katie drives be $x$. / Since Michelle drives  times the amount that Katie drives, $3x$. / And Tracy drives 2twice Michelle, which is $2(3x) + 2 = 6x + 2$ number of miles. / The total is 1000 miles, so we have x + 3x + x + 2 = 1000. / 1000 = 5x + 2. / Subtracting 2 from both sides gives 98 = 5x. / Dividing both sides by 5 gives x = 196. / So Katie drives 196 miles. / Michelle drives 3  …

**2b. MoE from draft, rule (b)** — ❌ wrong · 155 passes over 17 blocks [17, 23, 17, 26, 28, 23, 5, 3, 4, 1, 1, 1, 1, 1, 2, 1, 1] (fewer than normal) · HIT LENGTH CAP · rep 0.15 · block 2: kept 0/32, correct 9/32, kept∧correct 0
> Let's denote the number of miles Katie drives be $x$. / Since Michelle drives 3 times the amount that Katie drives, Michelle drives $3x$ miles. / And since Tracy drives $ 20 miles more than twice Michelle, Tracy drives $2(3x) + 20 = 6x + 20$ miles. / The total distance of the road trip is the sum of the distances driven by Tracy, Michelle, and Kati, which is equal to $\boxed{1000}1000$ miles. So we have the equation: …

**3. Normal blank-block decode** — ✅ correct · 170 passes over 9 blocks [16, 22, 16, 21, 23, 16, 22, 20, 14] · stopped · rep 0.18
> Let's denote the number of miles Katie drives be $x$. / Since Michelle drives 3 times the amount that Katie drives, Michelle drives $3x$ miles. / And since Tracy drives $ 20 miles more than twice Michelle, Tracy drives $2(3x) + 20 = 6x + 20$ miles. / The total distance of the road trip is the sum of the distances driven by each person, so we have the equation: $x + 3x + 6x + 20 = 1000$. / Combining like terms gives u …

### GSM8K train #604  —  reference answer **8**

**1. Drafter, block 2:** the year ( the the half half of the year) and the second half of the year ( the the half half half of the year) and then compare the

**2a. MoE from draft, rule (a)** — ✅ correct · 207 passes over 10 blocks [15, 24, 30, 17, 26, 26, 23, 15, 26, 5] (MORE than normal) · stopped · rep 0.43 · block 2: kept 1/32, correct 0/32, kept∧correct 0
> solve this problem, we need to calculate the number of presents Santana buys in the first half of the year (January to June, including June) and the second half of the year (July to December half) of the year separately, and then find the difference. /  / In, the first half of the year, Santana has 3 brothers with birthdays in March. So she buys 3 presents in the first half of the year. /  / In the second half of the …

**2b. MoE from draft, rule (b)** — ❌ wrong · 79 passes over 17 blocks [17, 17, 28, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] (fewer than normal) · HIT LENGTH CAP · rep 0.97 · block 2: kept 0/32, correct 4/32, kept∧correct 0
> Santana has 7 brothers. Out of these, 3 have birthdays in March, 1 has a birthday in October, 1 has a birthday in November, and 2 were born in December. / In the first half of the year (January to June), Santana has 3 brothers with birthdays. / birthday birthday birthday birthday birthday birthday birthday birthday birthday birthday birthday birthday birthday birthday birthday birthday birthday birthday birthday birt …

**3. Normal blank-block decode** — ❌ wrong · 182 passes over 10 blocks [16, 16, 22, 28, 26, 20, 16, 16, 10, 12] · stopped · rep 0.35
> Santana has 7 brothers. Out of these, 3 have birthdays in March, 1 has a birthday in October, 1 has a birthday in November, and 2 were born in December. / In the first half of the year (January to June), there are no birthdays for Santana's brothers. So she has to buy no presents in this period. / In the second half of the year (July to December), there are 1 birthdays in October, 1 birthday in November, and another  …

### GSM8K train #657  —  reference answer **730**

**1. Drafter, block 2:** He train ride is a 1  2 train to LA. /  which means he is 900 hours. / To 900 hours

**2a. MoE from draft, rule (a)** — ❌ wrong · 143 passes over 7 blocks [9, 25, 26, 22, 17, 31, 13] (MORE than normal) · stopped · rep 0.25 · block 2: kept 0/32, correct 3/32, kept∧correct 0
> Pete needs to take a 10-minute walk down to the train station. / He then boards a 1 hour and 20-minute train ride to LA, which is equivalent to 80 minutes (1 hour = 60 minutes + 60 + 120 minutes = a total of 80 minutes). / So in total, Pete needs 10 minutes for the + 80 minutes for the train ride = a total of 90 minutes. / To convert this into hours, we divide by 60: 0 minutes / 60 = 1.5 hours. / Pete wants to arrive …

**2b. MoE from draft, rule (b)** — ❌ wrong · 137 passes over 17 blocks [9, 25, 23, 23, 29, 15, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1] (MORE than normal) · HIT LENGTH CAP · rep 0.07 · block 2: kept 0/32, correct 3/32, kept∧correct 0
> Pete needs to take a 10-minute walk down to the train station. / He then boards a 1 hour and 20-minute train ride to LA, which is equivalent to 80 minutes (1 hour = 60 minutes, so 1 hour and 20 minutes = 60 + 20 = 80 minutes). / So in total, Pete will spend 10 + 80 = 90 minutes traveling. / To convert this into hours, we divide by 60 (since 0900 hours is 9 hours), so he will need to spend 90 / 60 = 1.5 hours. / If Pe …

**3. Normal blank-block decode** — ❌ wrong · 128 passes over 7 blocks [8, 24, 22, 23, 18, 19, 14] · stopped · rep 0.12
> Pete needs to take a 10-minute walk down to the train station. / He then boards a 1 hour and 20-minute train ride to LA, which is equivalent to 80 minutes (1 hour = 60 minutes, so 1 hour and 20 minutes = 60 + 20 = 80 minutes). / So in total, Pete will spend 10 + 80 = 90 minutes traveling from his home to LA. / To convert this into hours, we divide by 60 since there are 60 minutes in an hour: 90/60 = 1.5 hours. / If P …

### GSM8K train #659  —  reference answer **50**

**1. Drafter, block 2:** is $ola.00, she she makes 20 bags, $ola.00 x 2.00  $2000

**2a. MoE from draft, rule (a)** — ❌ wrong · 102 passes over 17 blocks [20, 15, 23, 27, 3, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1] (fewer than normal) · HIT LENGTH CAP · rep 0.28 · block 2: kept 3/32, correct 2/32, kept∧correct 0
> First let's find out Elizabeth's total cost of making granola. The total cost of ingredients for one bag $3.00, and she makes 20 bags, so3.00 * 2000 = $60.00. / She sells all 20 bags for $6.00 each, so her total revenue is $6.00 * 20 = $20.00. / Her net profit is her total profit minus her total cost, so it's $20.00 - $600.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 …

**2b. MoE from draft, rule (b)** — ❌ wrong · 50 passes over 17 blocks [13, 15, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1] (fewer than normal) · HIT LENGTH CAP · rep 0.04 · block 2: kept 10/32, correct 18/32, kept∧correct 4
> Elizabeth uses $3.00 worth of ingredients to make one bag of granola, and she makes 20 bags, so her total cost is $3.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 …

**3. Normal blank-block decode** — ❌ wrong · 133 passes over 7 blocks [12, 21, 20, 18, 20, 22, 20] · stopped · rep 0.22
> Elizabeth uses $3.00 worth of ingredients to make one bag of granola, and she makes 20 bags, so her total cost for ingredients is $3.00 x 20 = $60.00. / She sells each bag for $6.00, and she sells 15 bags at this price, so she earns 15 x $6.00 = $90.00 from these sales. / She then marks down the remaining 5 bags to $4.00 each, so she earns 5 x $4.00 = $20.00 from these sales. / Her total revenue is $600 + $90. + $20. …

### GSM8K train #674  —  reference answer **186**

**1. Drafter, block 2:** to be the the of of parking parking spots minus minus the the the number of open open spots on the the level minus minus the the the the level level the

**2a. MoE from draft, rule (a)** — ❌ wrong · 162 passes over 10 blocks [6, 32, 28, 20, 8, 10, 21, 7, 25, 5] (MORE than normal) · stopped · rep 0.40 · block 2: kept 0/32, correct 3/32, kept∧correct 0
> In this problem number of full parking spots is equal to the total number of parking spots minus the number of open parking spots. /  / The total number of parking spots is equal to the number of levels times the number of spots per level, which is $100 \times$ 4 = 400$. /  / The number of open parking spots on the first is given as 58. /  / The number of parking spots on the first level is 58, so the number of open  …

**2b. MoE from draft, rule (b)** — ❌ wrong · 191 passes over 17 blocks [7, 26, 23, 14, 26, 28, 32, 24, 1, 1, 2, 1, 1, 2, 1, 1, 1] (MORE than normal) · HIT LENGTH CAP · rep 0.45 · block 2: kept 0/32, correct 3/32, kept∧correct 0
> First, let's find out how many open parking spots there are on each level. / On the first level has 58 open spots. / So, the second level has 2 more open spots than the first level, which means it has 58 + 2 = 60 open spots. / And, the third level has 5 more open spots than the second level, which means it has 60 + 5 = 65 open spots. / Finally, the fourth level has 31 total open spots. / To find out how many full par …

**3. Normal blank-block decode** — ❌ wrong · 147 passes over 8 blocks [6, 25, 22, 13, 25, 22, 18, 16] · stopped · rep 0.28
> First, let's find out how many open parking spots there are on each level. / On the first level has 58 open spots. / So, the second level has 2 more open spots than the first level, which means it has 58 + 2 = 60 open spots. / And, the third level has 5 more open spots than the second level, which means it has 60 + 5 = 65 open spots. / Finally, the fourth level has 31 total open spots. / To find out how many full par …

### GSM8K train #735  —  reference answer **30**

**1. Drafter, block 2:** in an hour, so 10 minutes is equal to{1111111110001111111

**2a. MoE from draft, rule (a)** — ❌ wrong · 82 passes over 5 blocks [17, 14, 25, 23, 3] (fewer than normal) · stopped · rep 0.32 · block 2: kept 1/32, correct 12/32, kept∧correct 0
> First, we need to convert the time from minutes to hours since the speed is required in miles per hour. There are 60 minutes an hour, so 10 minutes is equal to $\10/60 = 1/6$ hours. / Next, we can use the formula speed = distance / time to find the speed Marcos needs to go. The distance is given as 5 miles / (1/6) = 5 miles *6/1 = 30$30 miles per hour. / So, the speed needs to be $\boxed{Marc to go is 30 per hour. /

**2b. MoE from draft, rule (b)** — ❌ wrong · 86 passes over 17 blocks [17, 21, 21, 9, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1] (fewer than normal) · HIT LENGTH CAP · rep 0.20 · block 2: kept 1/32, correct 12/32, kept∧correct 0
> First, we need to convert the time from minutes to hours since the speed is required in miles per hour. There are 60 minutes in an hour, so 10 minutes is equal to 10/60. / 10/60 = 1/6 hours. / Next, we can use the formula for speed, which is distance divided by time. The distance is 5 miles and the time is 1/6 hours. / Speed = 5.55555555555555555555555555555555555555555555555555555555555555555555555555555555555555555 …

**3. Normal blank-block decode** — ✅ correct · 87 passes over 6 blocks [16, 13, 18, 14, 19, 7] · stopped · rep 0.15
> First, we need to convert the time from minutes to hours since the speed is required in miles per hour. There are 60 minutes in an hour, so 10 minutes is $\frac{10}{60} = \frac{1}{6}$ hours. / Next, we can use the formula for speed, which is distance divided by time. The distance is 5 miles and the time is $\frac{1}{6}$ hours, so the speed is $\frac{5}{\frac{1}{6}} = 5 \times 6 = 30$ miles per hour. / Therefore, Marc …

### GSM8K train #747  —  reference answer **1500**

**1. Drafter, block 1:** John's rocket height rocket is the first of  000 feet. The second second rocket can twice  000 second0.

**2a. MoE from draft, rule (a)** — ❌ wrong · 65 passes over 16 blocks [28, 15, 5, 2, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1] (fewer than normal) · HIT LENGTH CAP · rep 0.28 · block 1: kept 2/32, correct 1/32, kept∧correct 0
> John's first rocket can travel 500 ft in the air0000000000000000 /  /  second rocket can travel a height twice as high as the first rocket, so it can travel 000000000000 / 00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 …

**2b. MoE from draft, rule (b)** — ❌ wrong · 47 passes over 16 blocks [29, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1] (fewer than normal) · HIT LENGTH CAP · rep 0.00 · block 1: kept 2/32, correct 1/32, kept∧correct 0
> John's first rocket travels 00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 …

**3. Normal blank-block decode** — ✅ correct · 72 passes over 4 blocks [23, 12, 20, 17] · stopped · rep 0.35
> John's first rocket can travel 500 ft in the air. / His second rocket can travel twice as high as the first one, so it can travel $2 \times 500 = 1000$ ft in the air. / To find the combined height of the two rockets, we add the heights of the two rockets together: $500 + 1,000 = 1500$. / So the combined height of the two rockets is $\boxed{1500}$ ft. /  /

### GSM8K train #750  —  reference answer **48**

**1. Drafter, block 2:** half dozenonsonsons and each crayon costs $2 per 2 cray * 2 = 2 * 2 = 2 = 2

**2a. MoE from draft, rule (a)** — ❌ wrong · 56 passes over 3 blocks [11, 26, 19] (fewer than normal) · stopped · rep 0.30 · block 2: kept 0/32, correct 0/32, kept∧correct 0
> Jamal bought the colored crayons Jamal bought 4 half dozen colored crayons per crayon at $2 per crayon. The total number of crayons is 4 * 6 = 24 crayons To find the total cost of the crayons, we multiply the number of crayons she bought (24) by the cost /  /  /

**2b. MoE from draft, rule (b)** — ✅ correct · 84 passes over 4 blocks [10, 26, 29, 19] (fewer than normal) · stopped · rep 0.32 · block 2: kept 0/32, correct 3/32, kept∧correct 0
> Jamal bought 4 half dozens of colored crayons. A half dozen is 6 crayons, so 4 half dozens is 4 * 6 = 24 crayons. / Each crayon costs $2, so the total cost of the crayons that she bought is 24 * = 24 * $2 24 * $2 = $48. / The answer is $\boxed{48}$

**3. Normal blank-block decode** — ✅ correct · 92 passes over 4 blocks [9, 25, 28, 30] · stopped · rep 0.30
> Jamal bought 4 half dozens of colored crayons. A half dozen is 6 crayons, so 4 half dozens is 4 * 6 = 24 crayons. / Each crayon costs $2, so the total cost of the crayons that she bought is 24 * = 24 * $2 24 * $2 = $48$ $48$. The answer is $\boxed{48} $\boxed{48}$. /

### GSM8K train #764  —  reference answer **5**

**1. Drafter, block 2:** food she had4 for lunch was  calories0.., so amount of of of  caloriesths of calories calories calories calories amount of calories calories calories calories

**2a. MoE from draft, rule (a)** — ❌ wrong · 61 passes over 5 blocks [8, 23, 12, 15, 3] (fewer than normal) · stopped · rep 0.32 · block 2: kept 0/32, correct 9/32, kept∧correct 0
> The dietitian ate three-fourths of her lunch, and the total amount of food she had prepared for lunch had 40 calories. So she ate (3/4) * 40 = 30 calories. / The recommended intake by the FDA is 25 calories. To find out how many more calories than the recommended amount she ate, we subtract the recommended amount from the amount ate: 30 - 25 = 5. / She ate 5 more calories than the recommended amount she ate. / The an …

**2b. MoE from draft, rule (b)** — ✅ correct · 64 passes over 5 blocks [8, 23, 13, 16, 4] (fewer than normal) · stopped · rep 0.32 · block 2: kept 0/32, correct 9/32, kept∧correct 0
> The dietitian ate three-fourths of her lunch, and the total amount of food she had prepared for lunch had 40 calories. So she ate (3/4) * 40 = 30 calories. / The recommended calorie intake by the FDA is 25 calories. To find out how many more calories than the recommended amount she ate, we subtract the recommended amount from the amount she ate: 30 - 25 = 5 calories. / So the dietitian ate 5 more calories than the re …

**3. Normal blank-block decode** — ✅ correct · 71 passes over 5 blocks [7, 22, 12, 15, 15] · stopped · rep 0.32
> The dietitian ate three-fourths of her lunch, and the total amount of food she had prepared for lunch had 40 calories. So she ate (3/4) * 40 = 30 calories. / The recommended calorie intake by the FDA is 25 calories. To find out how many more calories than the recommended amount she ate, we subtract the recommended amount from the amount she ate: 30 - 25 = 5 calories. / So the dietitian ate 5 more calories than the re …

### GSM8K train #776  —  reference answer **40**

**1. Drafter, block 2:** find out many hours hours worked worked by Armand. Ella worked for 12 hours and Armand worked one-third as long as Amber, so worked worked

**2a. MoE from draft, rule (a)** — ❌ wrong · 73 passes over 17 blocks [5, 21, 21, 5, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2] (fewer than normal) · HIT LENGTH CAP · rep 0.97 · block 2: kept 0/32, correct 3/32, kept∧correct 0
> First, let's out how many hours Amber worked. Armand worked one-third of 12 hours. Armand worked one-third as long as Amber. Armand worked 1/3 * 12 = 4 hours. Ella worked twice as long as Amber. Ella worked Ella Ella Ella Ella Ella Ella Ella Ella as long as Ella Ella Ella Ella Ella Ella Ella Ella Ella Ella Ella Ella Ella Ella Ella Ella Ella Ella Ella Ella Ella Ella Ella Ella Ella Ella Ella Ella Ella Ella Ella Ella El …

**2b. MoE from draft, rule (b)** — ❌ wrong · 158 passes over 8 blocks [5, 28, 30, 21, 13, 26, 22, 13] (fewer than normal) · stopped · rep 0.32 · block 2: kept 0/32, correct 3/32, kept∧correct 0
> First, let's find out how many hours that Armand worked. We know that Armand worked one-third as long as Amber, so we can divide the number of hours that Amber worked by 3.  /  / The number of hours that Armand worked 12 hours / 3  /  / So Amber worked for 12 Armand worked for 4 hours. /  / Next, let's find out how many hours that Ella worked. We know that Ella worked twice as long as Amber, so we can multiply the nu …

**3. Normal blank-block decode** — ✅ correct · 189 passes over 9 blocks [4, 27, 29, 20, 12, 25, 21, 25, 26] · stopped · rep 0.40
> First, let's find out how many hours that Armand worked. We know that Armand worked one-third as long as Amber, so we can divide the number of hours that Amber worked by 3.  /  / The number of hours that Armand worked 12 hours / 3  /  / So Amber worked for 12 Armand worked for 4 hours. /  / Next, let's find out how many hours that Ella worked. We know that Ella worked twice as long as Amber, so we can multiply the nu …

