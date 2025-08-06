
**The Challenge: Predicting Algorithm Performance**
Choosing the right algorithm for a complex task can save significant time and computational resources. However, many state-of-the-art algorithms are randomized, meaning their runtime isn't a single fixed number. One run might be incredibly fast, while the next, on the same problem, could be much slower.
To truly understand an algorithm's performance, we need to predict its entire Runtime Distribution (RTD). This allows us to see the full picture: from the best-case and worst-case runtimes to the most likely outcomes.

**Our Research**
This work investigates whether a new, general-purpose model can effectively solve this specialized prediction problem. We compare two powerful approaches:
The Specialist (DistNet): A deep learning model from Eggensperger et al. (2018) that was specifically designed to predict algorithm runtime distributions.
The Challenger (TabPFN v2): A powerful, pre-trained model designed for a wide variety of tabular data problems..

**Goal**
Evaluate how well TabPFN v2 can model the uncertainty and variability of algorithm runtimes, given a set of problem-specific features, and to see how its predictions stack up against the specialized DistNet approach.

**Distnet Baseline Reproduction**
We began by replicating the benchmark performance of the original DistNet model.
Our results successfully align with the Negative Log-Likelihood scores reported for the main network in Table 3 of Eggensperger et al. (2017).


**DISTNET PAPER:** https://arxiv.org/pdf/1709.07615


**Dataset:** http://www.ml4aad.org/wp-content/uploads/2018/04/DistNetData.zip

**TabPFN Pipeline File:** TabPFN_Pipeline.ipynb

**Conclusion**: Despite its significant memory constraints, the general-purpose TabPFN model shows competitive and often superior performance against the specialized DistNet for predicting algorithm runtime distributions.
