---
title: "ArXiv Daily Digest on 2025-10-06"
collection: digests
type: "ArXiv daily digest"
permalink: /digests/arxiv_cs_CL_2025-10-06_report
date: 2025-10-06
location: "Online"
---

Today's research landscape reveals a strong focus on enhancing large language models through sophisticated optimization techniques and multi-agent collaboration frameworks. A prominent trend involves addressing fundamental challenges in preference optimization, with papers like BVPO tackling gradient variance in large reasoning models through bias-variance trade-off optimization. The multi-agent paradigm continues to gain momentum, evidenced by MATPO's approach to training distinct agent roles within a single model and LLM-Hanabi's benchmark for evaluating theory-of-mind capabilities. Meanwhile, efficiency remains a critical concern, as demonstrated by MIFO's solution to catastrophic forgetting between supervised and reinforcement learning, achieving state-of-the-art results with dramatically reduced data requirements. These developments collectively push toward more stable, efficient, and collaborative AI systems.

## TL;DR

Here's a TL;DR summary of the key themes and insights from these papers:

**Core Focus:** Recent advances in improving language model training efficiency and multi-agent collaboration.

**Key Themes:**

1. **Reinforcement Learning Optimization** - Multiple papers address RL challenges: BVPO reduces gradient variance in reasoning models by mixing trace-based estimators; Reinforce-Ada uses adaptive sampling to prevent signal collapse; MATPO enables multi-agent RL training within a single model.

2. **Training Efficiency** - Several works focus on data efficiency: MIFO combines SFT and RL with only 1.5% SFT data and 20.4% RL data of previous SOTA, using parameter freezing to prevent catastrophic forgetting.

3. **Multi-Agent Collaboration** - LLM-Hanabi shows Theory-of-Mind (especially 1st-order reasoning) correlates with better collaboration; ACE evolves contexts as playbooks for self-improving agents.

4. **Architecture Insights** - Multilingual routing in MoE models reveals U-shaped patterns, with middle layers being language-universal; interventions that promote cross-lingual expert sharing improve performance.

**Main Insight:** There's a strong push toward more efficient, stable training methods that preserve learned capabilities while enabling complex multi-agent interactions, with particular attention to reasoning tasks and multilingual generalization.

---

# From Noisy Traces to Stable Gradients: Bias-Variance Optimized Preference Optimization for Aligning Large Reasoning Models

Authors: Mingkang Zhu, Xi Chen, Bei Yu, Hengshuang Zhao, Jiaya Jia

Keywords: Reinforcement Learning, Adaptive Sampling, Language Models, Policy Optimization, Training Efficiency

Comments: None

Paper link: http://arxiv.org/abs/2510.05095v1

## Abstract

Large reasoning models (LRMs) generate intermediate reasoning traces before producing final answers, yielding strong gains on multi-step and mathematical tasks. Yet aligning LRMs with human preferences, a crucial prerequisite for model deployment, remains underexplored. The statistically correct objective for preference alignment requires marginalizing over reasoning traces, but this computation is intractable in practice. A common workaround optimizes a single sampled trajectory, which introduces substantial gradient variance from stochastic trace sampling. To address this challenge, we frame preference optimization for LRMs through the lens of the bias--variance trade-off and propose Bias--Variance Optimized Preference Optimization (BVPO), a simple, drop-in method that mixes two gradient estimators: a high-variance trace-based estimator and a low-variance empty-trace estimator obtained by disabling reasoning trace generation. Our theory shows that BVPO strictly reduces trace-induced variance for any nontrivial mixture, provides a closed-form choice of the mixing weight that minimizes mean-squared error relative to the true marginal gradient, and under standard smoothness and step-size conditions, tightens classical convergence bounds for stochastic gradient descent. Empirically, BVPO improves alignment over the best baseline by up to 7.8 points on AlpacaEval~2 and 6.8 points on Arena-Hard. Despite being trained only on general conversational data, BVPO also boosts reasoning performance for base models by up to 4.0 points on the average of six math reasoning benchmarks. These results identify variance from trace sampling as a key bottleneck and demonstrate that directly optimizing the bias--variance trade-off yields more stable training and stronger overall performance.

## Summary

Based on the provided paper "Reinforce-Ada: An Adaptive Sampling Framework for Reinforce-Style LLM Training," here is a summary focusing on its key contributions, methods, and results:

**Key Contributions:**
The paper introduces Reinforce-Ada, an adaptive sampling framework designed to address the signal collapse problem in reinforcement learning for large language models (LLMs). The core issue is that standard methods like GRPO (Group Relative Policy Optimization) often produce zero gradients when all responses for a prompt yield identical rewards (either all correct or all incorrect), wasting computational resources and hindering learning. Reinforce-Ada dynamically allocates inference budget across prompts, sampling more responses for uncertain or challenging prompts while terminating early for easier ones, thereby improving training efficiency and stability.

**Methods:**
Reinforce-Ada operates through an online, multi-round sampling process inspired by successive elimination in multi-armed bandits. Key components include:
1. **Adaptive Sampling**: Prompts are sampled iteratively, and exit conditions deactivate them once sufficient signal is collected (e.g., Reinforce-Ada-pos stops after one correct response; Reinforce-Ada-balance requires a balanced mix of correct/incorrect responses).
2. **Global Normalization**: Advantages are computed using statistics from all collected responses per prompt, ensuring robust gradient estimates.
3. **Balanced Group Construction**: Responses are downsampled to fixed-size groups with enforced reward diversity to maintain non-zero gradients and training stability.

The framework integrates seamlessly into existing RL pipelines (e.g., GRPO) without architectural changes, acting as a plug-and-play replacement for uniform sampling.

**Results:**
Experiments across multiple LLMs (e.g., Qwen2.5-Math, Llama-3.2) and reasoning benchmarks (MATH500, Minerva Math) demonstrate that Reinforce-Ada:
- Accelerates convergence and achieves higher final rewards compared to GRPO, with Reinforce-Ada-balance yielding the best performance (+1–3 points in average accuracy).
- Improves sample efficiency and preserves policy diversity, leading to better trade-offs between reward and entropy and higher Pass@k scores at practical inference budgets.
- Shows greater benefits on challenging prompt sets, where adaptive allocation prevents signal loss more effectively. The computational overhead is moderate (1.4–2.8× step time), justified by significant performance gains.

Overall, Reinforce-Ada offers a scalable solution to enhance RL training for LLMs by optimizing data collection dynamically, bridging the gap between inference cost and learning signal quality.

## Critique

Of course. Here is a commentary on the strengths and weaknesses of the paper "Reinforce-Ada: An Adaptive Sampling Framework for Reinforce-Style LLM Training".

### Overall Assessment
This is a strong, well-executed paper that addresses a clear and important problem in RL for LLMs. The proposed method is practical, the empirical evaluation is thorough, and the results are significant. It makes a compelling case for adaptive sampling as a key component of efficient RL training pipelines.

---

### Strengths

**1. Clear Problem Formulation and Motivation:**
The paper expertly identifies a critical bottleneck in widely-used RL methods like GRPO: the "signal collapse" problem where fixed, small sample sizes lead to zero-gradient updates for many prompts. The introduction effectively uses data (Figure 2) to demonstrate that this is a statistical undersampling artifact, not a model limitation, which powerfully motivates the need for an adaptive solution.

**2. Practical and Novel Approach:**
The core idea of Reinforce-Ada—dynamically allocating more samples to prompts based on uncertainty in an online, successive-elimination manner—is both novel and highly practical.
*   **Novelty:** While related to bandit algorithms and prior work on budget allocation, its integration into the online RL loop for LLMs and its specific "exit condition" design is a distinct contribution. The move away from a two-stage "explore-then-exploit" paradigm to a unified online process is a key differentiator.
*   **Practicality:** The framework is presented as a "plug-and-play" replacement, requiring minimal changes to existing codebases (emphasized in Figure 1). This lowers the barrier to adoption and increases its potential impact.

**3. Thorough and Convincing Empirical Evaluation:**
The experimental section is a major strength.
*   **Scope:** The authors test across multiple model families (Qwen, Llama) and scales (1.5B to 7B), demonstrating the generality of their approach.
*   **Metrics:** Evaluation goes beyond simple accuracy, including analysis of training dynamics, reward-entropy trade-offs, and Pass@k curves. This provides a much richer picture of how the method affects model behavior.
*   **Ablation Studies:** The comparison between the "pos" and "balance" variants, along with the analysis of computational overhead and the impact of prompt set difficulty, offers deep insights into the method's behavior and trade-offs.

**4. Significant and Robust Results:**
The results are not just statistically significant but also practically so. Consistent gains of +1-3 points across diverse and challenging benchmarks (especially the aggregated AIME-like set) are highly meaningful in the context of mathematical reasoning. The finding that Reinforce-Ada provides even larger gains on more difficult prompt sets is a powerful point that underscores its value.

**5. Clear and Well-Structured Presentation:**
The paper is well-written and logically organized. The algorithm is clearly outlined, the figures are informative and directly support the narrative, and the relationship to prior work is comprehensively discussed.

---

### Weaknesses

**1. High Computational Overhead:**
The most significant weakness is the substantial increase in computational cost per training step (1.4x to 2.8x). While the paper rightly argues that this cost is justified by improved sample efficiency and final performance, it remains a practical limitation, especially for very large-scale training. A more detailed discussion on the total computational budget (steps × time-per-step) to reach a certain performance level would have been valuable.

**2. Limited Exploration of Exit Conditions:**
The paper proposes two exit conditions ("pos" and "balance") but their selection feels somewhat heuristic. While the balance variant is shown to be superior, the paper does not explore a more continuous or theoretically grounded condition (e.g., based on the variance of the advantage estimates or a confidence interval around the pass rate). This area seems ripe for further refinement.

**3. Domain Specificity:**
All experiments are confined to mathematical reasoning. While this is a canonical and challenging domain for RL, the paper's claims of generality would be stronger if it included results on a different modality, such as code generation or a general instruction-following task. It's unclear how the method would perform when rewards are non-binary or more noisy.

**4. Under-explored Synergy with Prompt Selection:**
The paper briefly mentions in Section 3.2 and the discussion that adaptive sampling would benefit from being combined with macro-level prompt selection strategies (curriculum learning). However, this is not experimentally demonstrated. A joint ablation studying the interaction between adaptive sampling and prompt difficulty scheduling could have been a valuable addition.

### Summary

**Reinforce-Ada** is a well-motivated, empirically-validated, and practical solution to a recognized problem in RL for LLMs. Its main strength lies in its effective and intuitive design that delivers consistent performance improvements. The primary trade-off is computational cost, which the results generally justify. The paper makes a solid contribution to the field and is likely to influence both future research and practical implementations of RL training pipelines. The weaknesses are not fatal but rather point to natural and promising directions for future work.

---

# Reinforce-Ada: An Adaptive Sampling Framework for Reinforce-Style LLM Training

Authors: Wei Xiong, Chenlu Ye, Baohao Liao, Hanze Dong, Xinxing Xu, Christof Monz, Jiang Bian, Nan Jiang, Tong Zhang

Keywords: Large Reasoning Models, Preference Optimization, Bias-Variance Trade-off, Gradient Estimation, Trace Sampling, Direct Preference Optimization, Alignment

Comments: 16 pages, 6 figures

Paper link: http://arxiv.org/abs/2510.04996v1

## Abstract

Reinforcement learning applied to large language models (LLMs) for reasoning tasks is often bottlenecked by unstable gradient estimates due to fixed and uniform sampling of responses across prompts. Prior work such as GVM-RAFT addresses this by dynamically allocating inference budget per prompt to minimize stochastic gradient variance under a budget constraint. Inspired by this insight, we propose Reinforce-Ada, an adaptive sampling framework for online RL post-training of LLMs that continuously reallocates sampling effort to the prompts with the greatest uncertainty or learning potential. Unlike conventional two-stage allocation methods, Reinforce-Ada interleaves estimation and sampling in an online successive elimination process, and automatically stops sampling for a prompt once sufficient signal is collected. To stabilize updates, we form fixed-size groups with enforced reward diversity and compute advantage baselines using global statistics aggregated over the adaptive sampling phase. Empirical results across multiple model architectures and reasoning benchmarks show that Reinforce-Ada accelerates convergence and improves final performance compared to GRPO, especially when using the balanced sampling variant. Our work highlights the central role of variance-aware, adaptive data curation in enabling efficient and reliable reinforcement learning for reasoning-capable LLMs. Code is available at https://github.com/RLHFlow/Reinforce-Ada.

## Summary

Based on the provided paper, here is a summary focusing on its key contributions, methods, and results:

**Key Problem and Contribution:** This paper addresses the challenge of aligning Large Reasoning Models (LRMs) with human preferences. LRMs, which generate intermediate reasoning traces before producing final answers, suffer from high gradient variance when standard preference optimization methods (like DPO) are applied, because these methods rely on sampling a single reasoning trace. The authors propose **Bias–Variance Optimized Preference Optimization (BVPO)**, a method designed to stabilize training by explicitly optimizing the bias-variance trade-off of the gradient estimator.

**Proposed Method (BVPO):** The core of BVPO is a convex combination of two gradient estimators:
1.  A **high-variance trace-based estimator (`g_t`)**, which is the standard estimator that uses a single sampled reasoning trace.
2.  A **low-variance empty-trace estimator (`g_e`)**, a novel estimator computed by conditioning the model on an empty trace (`r = ∅`), making it deterministic with respect to trace sampling.

The combined gradient is `g_c(α) = α * g_t + (1-α) * g_e`. The authors provide a theoretical framework for choosing the mixing coefficient `α` to minimize the Mean Squared Error (MSE) of the gradient estimate relative to the ideal (but intractable) marginal gradient.

**Theoretical Results:** The paper provides strong theoretical guarantees for BVPO:
*   It **provably reduces the conditional variance** induced by trace sampling.
*   It derives a **closed-form, MSE-optimal mixing coefficient** `α`, guaranteeing that the combined estimator performs no worse than the best individual estimator.
*   It connects this statistical optimality to **tighter convergence bounds for Stochastic Gradient Descent (SGD)**.

**Empirical Results:** Extensive experiments on models like DeepSeek-R1 demonstrate that BVPO significantly outperforms strong baselines (DPO and SimPO):
*   **Alignment Performance:** BVPO improved win rates by up to **7.8 points on AlpacaEval 2** and **6.8 points on Arena-Hard**.
*   **Reasoning Performance:** Crucially, despite being trained only on general conversational data, BVPO also **enhanced reasoning capabilities**, boosting the average performance on six math reasoning benchmarks (including AIME, MATH-500, and Minerva) by up to **4.0 points**.

In conclusion, this work identifies and mitigates a key bottleneck—trace-induced gradient variance—in aligning LRMs, resulting in a method that delivers superior alignment and even improves the core reasoning abilities of the models.

## Critique

Of course. Here is a balanced assessment of the paper's strengths and weaknesses.

### Summary

This paper, "From Noisy Traces to Stable Gradients," introduces **Bias–Variance Optimized Preference Optimization (BVPO)**, a method designed to improve the alignment of Large Reasoning Models (LRMs) with human preferences. The core problem is that the theoretically correct way to align LRMs involves an intractable marginalization over all possible reasoning traces. The standard workaround—using a single sampled trace—introduces high gradient variance, leading to unstable training. BVPO addresses this by combining the high-variance gradient from a sampled trace with a low-variance (but potentially biased) gradient from an "empty trace," optimizing the trade-off to minimize Mean Squared Error (MSE).

---

### Strengths

1.  **High Novelty and Clear Problem Identification:** The paper excels at identifying a specific, underexplored, and practically significant problem: the high gradient variance inherent in aligning modern LRMs (like DeepSeek R1, GPT-o1) due to their stochastic, lengthy reasoning traces. It convincingly argues why existing methods (DPO, SimPO) are suboptimal for this new class of models, carving out a clear niche for its contribution.

2.  **Elegant and Principled Solution:** The proposed method, BVPO, is conceptually simple, elegant, and grounded in classical statistical theory (bias-variance trade-off). The idea of mixing a high-variance and a low-variance estimator is intuitive, and the paper provides a strong theoretical framework for choosing the optimal mixing coefficient.

3.  **Strong and Cohesive Theoretical Foundation:** The paper is not just an empirical demonstration; it is backed by rigorous theoretical analysis. The proofs for variance reduction, MSE-optimal combination, and the direct link to tighter SGD convergence bounds are significant strengths. Theorem 4, which connects the statistically optimal estimator to the algorithmically optimal one for SGD, is a particularly compelling result that unifies the theory and practice.

4.  **Comprehensive and Compelling Empirical Validation:** The experimental results are extensive and convincing. The paper demonstrates improvements not on one but three different model sizes, across two major alignment benchmarks (Arena-Hard, AlpacaEval 2), and in two distinct operational modes ("Thinking" and "NoThinking"). Crucially, it also shows that alignment with general conversational data does not harm—and can even improve—performance on six diverse mathematical reasoning benchmarks. This addresses a critical concern for deploying aligned LRMs.

5.  **Excellent Clarity and Structure:** The paper is exceptionally well-written and structured. It follows a logical flow: problem identification, proposed solution, theoretical analysis, and empirical validation. The use of clear notation, well-defined loss functions, and a dedicated "Practical Implementation" subsection makes the method easy to understand and replicate.

---

### Weaknesses

1.  **Limited Exploration of the Optimal `α`:** While the theory provides a closed-form solution for the optimal mixing coefficient `α*`, the empirical section appears to treat `α` as a hyperparameter. It is unclear if the theoretically derived `α*` was used or if it was found empirically. A discussion or experiment validating the practical estimation or performance of the theoretical `α*` would strengthen the connection between theory and practice.

2.  **Ablation Studies and Sensitivity Analysis:** The paper would be strengthened by a more thorough ablation study. Key questions remain:
    *   How sensitive is the performance to the value of `α`? Is there a wide effective range, or is performance brittle?
    *   What is the individual contribution of the empty-trace loss? How does BVPO compare to using *only* the empty-trace loss (`α=0`), especially on reasoning tasks?
    *   While the variance of the trace-based gradient is empirically demonstrated in the appendix, a plot showing the training loss curve or gradient norm stability of BVPO versus baselines would provide direct, visual evidence of the claimed stabilization.

3.  **Narrow Scope of Baselines:** The empirical comparisons are limited to DPO and SimPO. While these are strong baselines, comparing against other recent preference optimization methods (like KTO, ORPO) or a simple baseline like using multiple trace samples to reduce variance (if computationally feasible) would provide a more complete picture of BVPO's standing.

4.  **Assumption of Empty Trace as a Valid Proxy:** The method relies on the assumption that the "empty-trace" conditioning provides a useful, low-bias signal. The paper could do more to discuss or validate this. In some domains, the reasoning trace might be so integral to the answer that conditioning on an empty trace creates a fundamentally different and misleading distribution, potentially increasing bias more than the theory assumes.

### Conclusion

This is a high-quality paper that makes a significant contribution to the field of aligning large language models. It identifies a timely and important problem, proposes a novel, simple, and theoretically-grounded solution, and backs it up with extensive experiments that demonstrate clear improvements in both alignment quality and reasoning capability. The main weaknesses lie in a slightly under-explored empirical analysis of the method's hyperparameters and ablations. Nonetheless, the core idea is powerful, the execution is thorough, and the paper is exceptionally clear and well-argued. It is likely to influence both future research and practical implementations of LRM alignment.

---

# LLM-Hanabi: Evaluating Multi-Agent Gameplays with Theory-of-Mind and Rationale Inference in Imperfect Information Collaboration Game

Authors: Fangzhou Liang, Tianshi Zheng, Chunkit Chan, Yauwai Yim, Yangqiu Song

Keywords: Multi-Agent Collaboration, Theory-of-Mind, Rationale Inference, Imperfect Information Games, LLM Evaluation, Cooperative AI

Comments: EMNLP 2025 Wordplay

Paper link: http://arxiv.org/abs/2510.04980v1

## Abstract

Effective multi-agent collaboration requires agents to infer the rationale behind others' actions, a capability rooted in Theory-of-Mind (ToM). While recent Large Language Models (LLMs) excel at logical inference, their ability to infer rationale in dynamic, collaborative settings remains under-explored. This study introduces LLM-Hanabi, a novel benchmark that uses the cooperative game Hanabi to evaluate the rationale inference and ToM of LLMs. Our framework features an automated evaluation system that measures both game performance and ToM proficiency. Across a range of models, we find a significant positive correlation between ToM and in-game success. Notably, first-order ToM (interpreting others' intent) correlates more strongly with performance than second-order ToM (predicting others' interpretations). These findings highlight that for effective AI collaboration, the ability to accurately interpret a partner's rationale is more critical than higher-order reasoning. We conclude that prioritizing first-order ToM is a promising direction for enhancing the collaborative capabilities of future models.

## Summary

This paper introduces **LLM-Hanabi**, a novel benchmark designed to evaluate **Theory-of-Mind (ToM)** and **rationale inference** in large language models (LLMs) within a dynamic, multi-agent collaborative setting. The benchmark is built around the cooperative card game *Hanabi*, where players have imperfect information and must rely on interpreting sparse linguistic hints from teammates to succeed. The key contribution is an automated evaluation framework that measures both game performance and ToM capabilities, addressing a gap in existing benchmarks that often rely on static, text-based tasks.

The methodology involves translating game states into natural language prompts, allowing LLM-driven agents to interact and reason collaboratively. A central feature is the **ToM Evaluation System**, which extracts structured reasoning during gameplay: the *rationale* behind a hint (ground truth), the recipient's *first-order ToM* (interpretation of the hint), and the hinter's *second-order ToM* (prediction of how the hint will be interpreted). Post-game, an LLM-as-a-judge scores alignment between these statements to quantify ToM proficiency.

Results across a diverse set of LLMs and large reasoning models (LRMs) reveal several key findings. First, **LRMs consistently outperform standard LLMs** in both game scores and ToM performance, with models like Deepseek-R1 and GPT-4.1 achieving the highest results. Second, there is a **strong positive correlation between ToM scores and game success**, indicating that better rationale inference leads to more effective collaboration. Most notably, **first-order ToM** (accurately interpreting a partner's intent) shows a stronger correlation with performance than **second-order ToM** (predicting others' interpretations), suggesting that for effective AI collaboration, the ability to infer rationale directly is more critical than higher-order reasoning. These insights highlight the importance of prioritizing first-order ToM capabilities in future model development for enhanced multi-agent collaboration.

## Critique

Of course. Here is a critique of the paper "LLM-Hanabi: Evaluating Multi-Agent Gameplays with Theory-of-Mind and Rationale Inference," focusing on its strengths and weaknesses.

### Overall Assessment

This is a well-structured and timely paper that makes a clear and valuable contribution to the field of multi-agent AI. It successfully leverages a classic game environment to create a novel, automated benchmark for evaluating complex cognitive skills in Large Language Models (LLMs).

---

### Strengths

1.  **Novelty and Clever Benchmark Design:**
    *   The choice of **Hanabi** is excellent. Its core mechanics—imperfect information, cooperative goals, and constrained communication—naturally isolate and test the specific capabilities of interest: Theory-of-Mind (ToM) and rationale inference.
    *   The **automated evaluation system** is a significant strength. By prompting agents to generate structured reasoning statements (Rationale, 1st-order ToM, 2nd-order ToM) and using an LLM-as-a-judge to score their alignment, the authors create a scalable and quantitative framework that moves beyond static, text-based ToM tests.

2.  **Significant and Actionable Results:**
    *   The finding of a **strong positive correlation between ToM performance and game success** is a crucial empirical result. It provides concrete evidence that these cognitive abilities are not just abstract metrics but are directly tied to effective collaboration.
    *   The most impactful finding is the **differential importance of 1st-order vs. 2nd-order ToM**. Demonstrating that 1st-order ToM (interpreting a partner's intent) is a stronger predictor of success than 2nd-order ToM (predicting a partner's interpretation) offers a clear and practical direction for future model development. It suggests that resources might be better spent on improving basic inference rather than complex recursive reasoning.

3.  **Clarity and Comprehensive Evaluation:**
    *   The paper is very well-written. The introduction clearly outlines the research gap, and the contributions are stated explicitly.
    *   The evaluation is thorough, benchmarking a wide range of models (both LLMs and the newer LRMs) and presenting results clearly in Table 1 and Figure 2. The distinction between model classes and the performance trends are easy to follow.

---

### Weaknesses

1.  **Methodological Concerns with Evaluation:**
    *   **Reliance on LLM-as-a-Judge:** The entire ToM scoring mechanism hinges on the subjective judgment of another LLM. While pragmatic, this introduces a potential confounder. The authors acknowledge this in the limitations, but it remains a key weakness. The judge's own biases and capabilities could influence the scores, and the paper does not provide a validation of this method (e.g., against human judgments) to calibrate its reliability.
    *   **Potential for "Gaming the System":** The prompts used to elicit Rationale and ToM statements are critical. If these prompts lead the models toward generating the "right-sounding" answers rather than revealing their true internal reasoning, the evaluation could be measuring prompt-following rather than genuine cognitive ability. More details on prompt design and an analysis of potential artifacts would strengthen the methodology.

2.  **Limited Discussion of "How" and "Why":**
    *   The paper excellently establishes the *"what"* (a correlation exists) but provides less insight into the *"why"*. For instance, *why* do LRMs consistently outperform LLMs? Is it the architecture, the training data, or the long-context reasoning? A deeper discussion of the potential reasons behind the performance gap between model classes would add significant value.
    *   Similarly, the analysis of *why* 1st-order ToM is more critical, while compelling, is somewhat surface-level. A more detailed analysis of game scenarios where accurate 1st-order inference directly leads to success, versus cases where failed 2nd-order predictions cause problems, would make the argument more concrete.

3.  **Scope and Generalizability:**
    *   As noted in the limitations, the benchmark is confined to a single, turn-based, communication-heavy game. The extent to which these findings generalize to other collaborative settings (e.g., real-time environments, tasks with physical components, or scenarios with conflicting sub-goals) is an open question. The paper could do more to hypothesize about this generalizability.

### Summary

**Strengths:** Novel and well-designed benchmark, significant and actionable empirical findings, clear and comprehensive presentation.
**Weaknesses:** Reliance on a potentially biased automated evaluation method, a somewhat limited exploration of the underlying reasons for the observed results, and questions about generalizability beyond the Hanabi environment.

Despite the weaknesses, which are common in pioneering work, the paper's strengths make it a valuable contribution. It provides a much-needed tool for the community and offers a compelling, data-driven argument for prioritizing 1st-order rationale inference in the development of collaborative AI agents.

---

# Multilingual Routing in Mixture-of-Experts

Authors: Lucas Bandarkar, Chenyuan Yang, Mohsen Fayyaz, Junlin Hu, Nanyun Peng

Keywords: Reinforcement Learning, Supervised Fine-tuning, Catastrophic Forgetting, Reasoning Capability, Parameter Freezing, Data Efficiency

Comments: None

Paper link: http://arxiv.org/abs/2510.04694v1

## Abstract

Mixture-of-Experts (MoE) architectures have become the key to scaling modern LLMs, yet little is understood about how their sparse routing dynamics respond to multilingual data. In this work, we analyze expert routing patterns using parallel multilingual datasets and present highly interpretable layer-wise phenomena. We find that MoE models route tokens in language-specific ways in the early and late decoder layers but exhibit significant cross-lingual routing alignment in middle layers, mirroring parameter-sharing trends observed in dense LLMs. In particular, we reveal a clear, strong correlation between a model's performance in a given language and how similarly its tokens are routed to English in these layers. Extending beyond correlation, we explore inference-time interventions that induce higher cross-lingual routing alignment. We introduce a method that steers the router by promoting middle-layer task experts frequently activated in English, and it successfully increases multilingual performance. These 1-2% gains are remarkably consistent across two evaluation tasks, three models, and 15+ languages, especially given that these simple interventions override routers of extensively trained, state-of-the-art LLMs. In comparison, interventions outside of the middle layers or targeting multilingual-specialized experts only yield performance degradation. Altogether, we present numerous findings that explain how MoEs process non-English text and demonstrate that generalization is limited by the model's ability to leverage language-universal experts in all languages.

## Summary

This paper introduces MIFO (Mitigating Forgetting Between Supervised and Reinforcement Learning), a novel framework that addresses the challenge of catastrophic forgetting when combining supervised fine-tuning (SFT) and reinforcement learning (RL) for reasoning tasks in large language models. The authors identify that SFT tends to produce redundant parameter updates with large magnitudes, while RL updates are more parsimonious and crucial for reasoning performance. When SFT follows RL training, it often overwrites RL-acquired knowledge, leading to significant performance degradation.

The key contributions of MIFO include: (1) A plug-and-play framework that dynamically interleaves SFT and RL while mitigating forgetting; (2) Two core components: data processing that selects challenging examples for SFT based on RL rollout accuracy and focuses on high-entropy tokens, and parameter freezing that identifies and protects RL-critical parameters during SFT; (3) Achieving state-of-the-art performance with substantially reduced data requirements.

Experiments on mathematical reasoning benchmarks (AIME, AMC, OlympiadBench, MATH500, MMLU-Pro) using Qwen2.5-Math models (1.5B and 7B) demonstrate that MIFO outperforms existing methods including LUFFY and SRFT. Notably, MIFO achieves these results using only 1.5% of the SFT data and 20.4% of the RL data compared to previous SOTA methods, while also producing more concise reasoning traces. The framework maintains strong performance across different model sizes and template variations, showing robustness in practical deployment scenarios.

The work provides important insights into the complementary roles of SFT and RL in reasoning training, with SFT importing external knowledge and RL reshaping internal probability distributions, and offers an efficient solution to leverage both approaches without catastrophic interference.

## Critique

Of course. Here is a detailed commentary on the strengths and weaknesses of the paper "Mitigating Forgetting Between Supervised and Reinforcement Learning Yields Stronger Reasoners".

### Overall Summary

This is a strong, well-executed paper that tackles a practical and important problem in the post-training of Large Language Models (LLMs). The core idea—that Supervised Fine-Tuning (SFT) catastrophically forgets knowledge gained from Reinforcement Learning (RL) due to its larger, more redundant parameter updates—is compelling and well-supported by empirical evidence. The proposed method, MIFO, is elegant, effective, and achieves impressive results with high data efficiency.

---

### Strengths

**1. High Novelty and Insightful Core Contribution:**
The paper's primary strength is its novel and well-argued observation about the asymmetric relationship between SFT and RL. The claim that "SFT forgets RL" is counter to the standard SFT-then-RL training recipe and is backed by a clear, intuitive explanation (magnitude of updates) and solid experiments (Figure 3). The complementary finding that SFT updates are "redundant" while RL updates are "parsimonious" (Figures 1 & 2) is a significant insight that directly informs the design of the solution.

**2. Elegant and Well-Motivated Method:**
MIFO is not a complex, monolithic new algorithm but a clever "plug-and-play" framework built on two well-justified components:
*   **Data Processing:** Selecting challenging examples and high-entropy tokens for SFT is a smart way to focus learning and inherently reduce the scale of SFT updates.
*   **Parameter Freezing:** Dynamically identifying and freezing RL-critical parameters is a direct and effective mitigation to the core problem of forgetting. The use of a rolling history of parameter importance (`C_i`) is a sophisticated touch that accounts for long-term training stability.

**3. Significant and Comprehensive Results:**
The experimental results are a major strength. The paper demonstrates:
*   **State-of-the-Art Performance:** MIFO achieves top-tier results on a comprehensive suite of challenging mathematical reasoning benchmarks for both 1.5B and 7B models.
*   **Exceptional Data Efficiency:** The claim of using only **1.5% of the SFT data** and **20.4% of the RL data** of the previous SOTA is a monumental result, making advanced LLM post-training significantly more accessible.
*   **Response Length Efficiency:** MIFO models produce more concise reasoning traces, which is highly desirable for reducing computational costs during inference.
*   **Thorough Ablations:** The ablation study (Table 3) clearly demonstrates the contribution of each component (Interleaving, Entropy Selection, Parameter Freezing) and shows that they are complementary.

**4. Clarity and Presentation:**
The paper is generally well-written and structured. The motivation is clear, the method is explained with pseudocode and a helpful diagram (Figure 4), and the results are presented comprehensively. The inclusion of extensive appendices (theoretical analysis, hyperparameter studies, template ablations) adds rigor and depth.

---

### Weaknesses

**1. Limited Exploration of the "Why":**
While the paper excellently demonstrates the *phenomenon* of asymmetric updating, the theoretical explanation in Appendix C, while a positive addition, remains relatively high-level. A deeper analysis of *why* SFT gradients are inherently more redundant (e.g., is it the nature of the cross-entropy loss on a fixed dataset vs. the policy gradient on a dynamic distribution?) would have strengthened the foundational contribution. The provided theoretical analysis feels more like a formalization of the observation rather than a root-cause explanation.

**2. Narrow Problem Scope:**
The paper focuses exclusively on **mathematical reasoning**. While this is a valid and important domain, it is a specific type of task. The claim that SFT forgets RL might be less pronounced or manifest differently in other domains like creative writing, code generation, or open-ended dialogue. The generality of the core problem and the solution's effectiveness across a broader range of tasks remains to be proven.

**3. Overstated "Plug-and-Play" Claim:**
The framework is more "plug-and-play" than methods like LUFFY that are fused into a single objective. However, MIFO still introduces new, non-trivial hyperparameters: the entropy threshold `ρ`, the history coefficient `α`, the top-k parameter fraction, and the buffer threshold `p`. While the paper includes a hyperparameter analysis (Appendix E.1), successfully applying MIFO to a new RL algorithm or domain would still require careful tuning.

**4. Presentation Minor Issues:**
*   **Figure Readability:** Some figures (e.g., Figure 1, 2) are a bit small and their captions are cut off ("Refer to caption"), which slightly hinders readability.
*   **Result Tables:** The tables are dense. While the color-coding helps, a brief summary sentence in the main text highlighting the most significant comparisons (e.g., "MIFO outperforms LUFFY by X points on average while using X% of the data") would make the key takeaways more immediately accessible to the reader.

---

### Conclusion

This is a high-quality paper with a **novel and important insight** that leads to a **practically effective method**. The strengths far outweigh the weaknesses. The core observation of SFT-RL interference is likely to influence future research in LLM post-training. MIFO delivers what it promises: stronger reasoning models trained with remarkable data and inference efficiency. The main limitations are the scope of evaluation (only math) and the need for a more fundamental theoretical explanation, but these do not detract from the significance of the empirical findings and the practical utility of the proposed framework.

---

# Multi-Agent Tool-Integrated Policy Optimization

Authors: Zhanfeng Mo, Xingxuan Li, Yuntao Chen, Lidong Bing

Keywords: Mixture-of-Experts, Multilingual Routing, Cross-lingual Transfer, Expert Activation, Sparse Models, Language Universality, Inference-time Interventions, Router Steering

Comments: Work in progress

Paper link: http://arxiv.org/abs/2510.04678v1

## Abstract

Large language models (LLMs) increasingly rely on multi-turn tool-integrated planning for knowledge-intensive and complex reasoning tasks. Existing implementations typically rely on a single agent, but they suffer from limited context length and noisy tool responses. A natural solution is to adopt a multi-agent framework with planner- and worker-agents to manage context. However, no existing methods support effective reinforcement learning post-training of tool-integrated multi-agent frameworks. To address this gap, we propose Multi-Agent Tool-Integrated Policy Optimization (MATPO), which enables distinct roles (planner and worker) to be trained within a single LLM instance using role-specific prompts via reinforcement learning. MATPO is derived from a principled credit assignment mechanism across planner and worker rollouts. This design eliminates the need to deploy multiple LLMs, which would be memory-intensive, while preserving the benefits of specialization. Experiments on GAIA-text, WebWalkerQA, and FRAMES show that MATPO consistently outperforms single-agent baselines by an average of 18.38% relative improvement in performance and exhibits greater robustness to noisy tool outputs. Our findings highlight the effectiveness of unifying multiple agent roles within a single LLM and provide practical insights for stable and efficient multi-agent RL training.

## Summary

This paper presents a comprehensive analysis of multilingual routing patterns in Mixture-of-Experts (MoE) large language models and demonstrates how inference-time interventions can improve multilingual performance. The key contributions include revealing interpretable layer-wise routing phenomena and developing effective intervention methods that consistently enhance cross-lingual transfer.

The authors conduct extensive routing analysis using parallel multilingual datasets (FLoRes-200) across four prominent MoE LLMs (Qwen3, Phi-3.5-MoE, GPT-OSS, OLMoE). They discover a consistent U-shaped pattern in routing divergence from English across model layers, with significantly lower divergence in middle layers compared to early and late layers. This mirrors findings from dense LLMs about language-universal representation spaces in intermediate layers. Crucially, they find a strong correlation between a model's performance in a language and how similarly its tokens are routed to English in these middle layers.

Building on these observations, the authors develop intervention methods that manipulate router logits during inference. They identify task-specific experts (for math and medicine domains) using English data and then intervene to promote activation of these same experts when processing non-English inputs. The interventions come in two forms: soft interventions (steering router logits by adding/subtracting values proportional to logit standard deviation) and hard interventions (force-activating or deactivating specific experts).

The results demonstrate consistent improvements across three models (Qwen3, Phi-3.5-MoE, GPT-OSS) and two evaluation tasks (MGSM for math reasoning and Global-MMLU medicine subset). The interventions yield 1-2% average gains across 15+ languages, with particularly strong improvements for lower-resource languages (up to 4.5% improvement for Swahili in MGSM). The success is highly sensitive to target layers, working only when applied to the identified middle "language-universal" layers, validating the routing analysis findings. This work reveals that cross-lingual routing alignment is a key driver of multilingual generalization in MoE models and suggests promising directions for improving multilingual performance through enhanced expert sharing.

## Critique

Of course. Here is a critique of the paper "Multilingual Routing in Mixture-of-Experts," covering its strengths, weaknesses, novelty, significance, and clarity.

### Overall Summary

This is a strong, well-executed paper that provides valuable insights into the internal workings of multilingual Mixture-of-Experts (MoE) models. It successfully bridges the gap between interpretability research on dense models and the increasingly prevalent MoE architecture, delivering both compelling observational analysis and a practical, causal intervention.

---

### Strengths

1.  **Novelty and Bridging a Research Gap:** The core novelty lies in applying established multilingual interpretability techniques (e.g., analyzing layer-wise representations) to the distinct, sparse architecture of MoE models. The finding that MoEs exhibit a similar "U-shaped" pattern of language-specific vs. language-universal processing as dense models, but through expert routing, is a significant contribution.
2.  **Strong Causal Evidence:** The paper moves beyond correlation to causation through its intervention experiments. Demonstrating that performance can be improved by manually steering routers in the middle layers to mimic English activation patterns is a powerful result. This solidifies the hypothesis that cross-lingual routing alignment is a key driver of multilingual generalization.
3.  **Comprehensive and Rigorous Evaluation:** The study is thorough. It analyzes four different MoE models (Qwen3, Phi-3.5-MoE, GPT-OSS-20B, OLMoE), uses multiple parallel datasets (FLoRes-200, MGSM, Global-MMLU), and evaluates on two distinct tasks (mathematical reasoning and medical QA). This breadth makes the findings more robust and generalizable.
4.  **Clear and Actionable Findings:** The paper presents several clear, well-supported findings:
    *   The U-shaped routing divergence pattern.
    *   The correlation between performance and routing alignment in middle layers.
    *   The modular separation of language and task experts.
    *   The success of middle-layer, task-expert interventions.
5.  **Clear Methodology:** The explanations of the routing divergence metric, expert identification process (`Δ` values), and the two intervention types (soft and hard) are detailed and appear reproducible.

---

### Weaknesses

1.  **Modest Performance Gains:** While statistically significant and remarkably consistent, the performance improvements from interventions are modest (1-2%). The paper is transparent about this, but it does raise the question of the practical impact versus the computational overhead of implementing such router steering at inference time.
2.  **Limited Exploration of Intervention Specifics:** The paper does an excellent job identifying *where* to intervene (middle layers) and *what* to intervene on (task experts), but the exploration of *how* is less detailed. The choice of `λ` and `τ` values seems to be based on empirical tuning. A more systematic ablation of these hyperparameters could have strengthened the methodology.
3.  **English-Centric Perspective:** The entire analysis and intervention strategy are predicated on using English as the "pivot" language. While this is a practical choice given the training data of these models, it inherently reinforces English-centricity. It would be interesting to see if the same principles hold when using another high-resource language as the source for expert identification.
4.  **Speculative Observations:** A few observations, while interesting, are presented without a definitive explanation. For instance, the note that Phi-3.5-MoE activates the same few experts in the first layer for all languages is flagged as "perplexing." While it's okay to report such phenomena, it leaves a minor question unanswered.

---

### Significance of Results

The results are highly significant for the field of multilingual NLP and LLM interpretability.

*   **Interpretability:** They provide a new lens through which to understand how MoE models process multilingual data, confirming and extending findings from dense models into the sparse regime.
*   **Model Design:** The clear modularity between language and task experts provides a compelling argument for future architectural or training innovations that explicitly encourage this separation, potentially leading to more efficient and capable multilingual models.
*   **Potential for Improvement:** The intervention results are a proof-of-concept that multilingual performance in massive, fixed models is not yet optimal. This opens the door for future work on training-time objectives or more sophisticated inference-time methods to enhance cross-lingual expert sharing.

---

### Clarity of Presentation

The paper is exceptionally well-written and structured.

*   **Narrative Flow:** The progression from observational analysis to hypothesis formation to causal intervention is logical and easy to follow.
*   **Visualizations:** Figures 1 and 2 are excellent, providing an immediate, intuitive understanding of the core "U-shape" finding and its correlation with performance.
*   **Precision:** The language is precise, and key findings are explicitly numbered and highlighted, making the paper's contributions very clear.
*   **Reproducibility:** The methodology is described in sufficient detail, with formulas and thresholds provided. The appendix offers further necessary specifics.

### Conclusion

This is a high-quality paper that makes a substantial contribution by elucidating the multilingual routing dynamics of MoE LLMs. Its strengths in novelty, rigorous experimentation, and clear causal demonstrations far outweigh its minor weaknesses. The work not only advances our theoretical understanding but also provides a concrete, actionable direction for improving multilingual generalization in future LLMs.

---

# Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models

Authors: Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James Zou, Kunle Olukotun

Keywords: Multi-Agent Reinforcement Learning, Tool-Integrated Planning, Policy Optimization, Multi-Agent Systems, Language Agents

Comments: None

Paper link: http://arxiv.org/abs/2510.04618v1

## Abstract

Large language model (LLM) applications such as agents and domain-specific reasoning increasingly rely on context adaptation -- modifying inputs with instructions, strategies, or evidence, rather than weight updates. Prior approaches improve usability but often suffer from brevity bias, which drops domain insights for concise summaries, and from context collapse, where iterative rewriting erodes details over time. Building on the adaptive memory introduced by Dynamic Cheatsheet, we introduce ACE (Agentic Context Engineering), a framework that treats contexts as evolving playbooks that accumulate, refine, and organize strategies through a modular process of generation, reflection, and curation. ACE prevents collapse with structured, incremental updates that preserve detailed knowledge and scale with long-context models. Across agent and domain-specific benchmarks, ACE optimizes contexts both offline (e.g., system prompts) and online (e.g., agent memory), consistently outperforming strong baselines: +10.6% on agents and +8.6% on finance, while significantly reducing adaptation latency and rollout cost. Notably, ACE could adapt effectively without labeled supervision and instead by leveraging natural execution feedback. On the AppWorld leaderboard, ACE matches the top-ranked production-level agent on the overall average and surpasses it on the harder test-challenge split, despite using a smaller open-source model. These results show that comprehensive, evolving contexts enable scalable, efficient, and self-improving LLM systems with low overhead.

## Summary

Here is a concise summary of the paper "Multi-Agent Tool-Integrated Policy Optimization (MATPO)":

**Key Contributions**: This paper introduces MATPO, a novel reinforcement learning framework that enables multiple agent roles (planner and worker agents) to be trained within a single large language model instance. This addresses the limitations of single-agent tool-integrated planning systems, which suffer from context length constraints and noisy tool responses. The key innovation is a principled credit assignment mechanism that allows both planner and worker rollouts to share responsibility for the final accuracy.

**Methodology**: MATPO extends single-agent Group Relative Policy Optimization (GRPO) to the multi-agent setting by deriving a multi-agent policy gradient objective. The framework uses role-specific system prompts to activate different agent behaviors within the same model, maintaining infrastructure efficiency while enabling specialization. The implementation builds on existing RL frameworks and includes practical components like rollout summarization mechanisms and user query recapping to worker agents.

**Results**: Experiments on GAIA-text, WebWalkerQA, and FRAMES benchmarks show that MATPO consistently outperforms single-agent GRPO baselines by an average of 18.38% relative improvement. The method demonstrates greater robustness to noisy tool outputs and more stable training progress. Ablation studies confirm the importance of key design choices, including final summaries for worker agents and recapping original user queries to worker agents. The framework also shows resilience to potential data contamination from search APIs.

This work represents a significant step toward efficient multi-agent reinforcement learning training within unified model architectures, offering both theoretical foundations and practical implementation insights.

## Critique

Of course. Here is a critique of the paper "Multi-Agent Tool-Integrated Policy Optimization (MATPO)", covering its strengths, weaknesses, novelty, results, and presentation.

### Overall Summary

This paper presents a well-motivated and technically sound approach to a practical problem in agentic AI: training a single Large Language Model (LLM) to perform multiple, distinct roles (planner and worker) within a multi-agent system using reinforcement learning. The work is significant as it addresses key infrastructure and performance bottlenecks of existing methods.

---

### Strengths

1.  **Clear Problem Identification:** The paper effectively motivates its work by clearly outlining the limitations of single-agent Tool-Integrated Planning (TIP), specifically the context window bottleneck and the disruptive effect of noisy tool outputs. The proposed multi-agent solution is a natural and logical step.
2.  **Novelty and Technical Contribution:** The core idea of "multi-agent-in-one-model" is novel and addresses a critical infrastructure challenge. Deploying multiple specialized agents typically requires multiple model instances, which is computationally expensive. MATPO's approach of using role-specific prompts to enable a single model to act as both planner and worker is a clever and resource-efficient innovation.
3.  **Principled Formulation:** The paper provides a rigorous mathematical derivation of the MATPO objective, extending the established Group Relative Policy Optimization (GRPO) framework to the multi-agent, multi-rollout scenario. This includes a principled solution to the credit assignment problem, ensuring both planner and worker actions are updated based on the final task reward.
4.  **Strong Empirical Results:** The experimental results are compelling. MATPO demonstrates a consistent and significant performance improvement (average 18.38% relative gain) over the single-agent GRPO baseline across three diverse benchmarks (GAIA-text, WebWalkerQA, FRAMES). The observation that MATPO is more robust and avoids the performance collapse seen in single-agent training is a particularly strong point.
5.  **Valuable Practical Insights:** The "Ablation Studies and Practical Take-Aways" section is a major strength. It moves beyond simply presenting results to offer actionable advice for practitioners (e.g., the necessity of final summaries, the mild impact of blocking certain URLs, the importance of user query recapping). This greatly enhances the paper's practical utility.

### Weaknesses

1.  **Limited Baseline Comparison:** While the comparison to single-agent GRPO is solid, the paper would be strengthened by comparisons to other multi-agent baselines, even if they are inference-only or use multiple models. This would better contextualize MATPO's performance gains relative to the broader field of multi-agent systems.
2.  **Scalability and Cost Analysis:** The implementation, while elegant, involves nested and asynchronous rollouts. The paper does not discuss the computational overhead or latency compared to single-agent rollouts. A discussion of the trade-off between performance gains and increased inference/training cost would be valuable for assessing the method's practicality.
3.  **Ablation on the Core Idea:** The most critical ablation—comparing "multi-agent-in-one-model" (MATPO) to a "multi-agent-multi-model" setup—is missing. It remains an open question whether the performance gains come from the multi-agent structure itself or from the specific "in-one-model" training paradigm. This is a key point for validating the paper's central thesis.
4.  **Qualitative Analysis:** The paper relies heavily on quantitative accuracy scores. Including more qualitative examples of successful and failed rollouts, or a deeper analysis of *how* the planner and worker behaviors co-evolve during training, would provide richer insights into the method's inner workings.

### Novelty and Significance

*   **Novelty:** High. The formulation of a multi-agent RL objective for a single model instance is a distinct contribution. The theoretical extension of GRPO and the concrete implementation that allows a single LLM to learn and execute specialized roles through prompt-based context switching is a novel and non-obvious solution.
*   **Significance:** High. As AI agents tackle more complex, real-world tasks, efficient and robust multi-agent collaboration becomes essential. MATPO offers a path forward that is both performant and resource-efficient, avoiding the prohibitive cost of deploying many large models. The insights into training stability and system design are immediately valuable to researchers and engineers in the field.

### Clarity of Presentation

*   **Clarity:** Overall, the paper is well-structured and clearly written. The problem setup is logical, and the methodology section, while mathematically dense, is necessary and well-explained.
*   **Visualizations:** The figures (1, 2, 3, and 4) are excellent. They provide intuitive, high-level summaries of the single-agent, multi-agent, and MATPO training workflows, making the complex concepts accessible.
*   **Areas for Improvement:** The transition from the problem setup to the methodology could be slightly smoother for readers less familiar with policy gradient theorems. Explicitly stating the high-level intuition behind the credit assignment in MATPO before diving into the equations would improve readability.

### Conclusion

This is a strong paper that makes a valuable contribution to the field of AI agents and reinforcement learning. It identifies a clear problem, proposes a novel and principled solution, and backs it up with solid empirical evidence and highly practical insights. While it could be strengthened by more comprehensive baselines and a deeper analysis of computational trade-offs, its core ideas are significant and likely to influence future work on efficient and collaborative AI systems.

---

# Mitigating Forgetting Between Supervised and Reinforcement Learning Yields Stronger Reasoners

Authors: Xiangchi Yuan, Xiang Chen, Tong Yu, Dachuan Shi, Can Jin, Wenke Lee, Saayan Mitra

Keywords: Agentic Context Engineering, Context Adaptation, Self-Improving Language Models, Multi-Agent Systems, Incremental Delta Updates, Grow-and-Refine, LLM Memory Systems

Comments: None

Paper link: http://arxiv.org/abs/2510.04454v1

## Abstract

Large Language Models (LLMs) show strong reasoning abilities, often amplified by Chain-of-Thought (CoT) prompting and reinforcement learning (RL). Although RL algorithms can substantially improve reasoning, they struggle to expand reasoning boundaries because they learn from their own reasoning trajectories rather than acquiring external knowledge. Supervised fine-tuning (SFT) offers complementary benefits but typically requires large-scale data and risks overfitting. Recent attempts to combine SFT and RL face three main challenges: data inefficiency, algorithm-specific designs, and catastrophic forgetting. We propose a plug-and-play framework that dynamically integrates SFT into RL by selecting challenging examples for SFT. This approach reduces SFT data requirements and remains agnostic to the choice of RL or SFT algorithm. To mitigate catastrophic forgetting of RL-acquired skills during SFT, we select high-entropy tokens for loss calculation and freeze parameters identified as critical for RL. Our method achieves state-of-the-art (SoTA) reasoning performance using only 1.5% of the SFT data and 20.4% of the RL data used by prior SoTA, providing an efficient and plug-and-play solution for combining SFT and RL in reasoning post-training.

## Summary

Based on the provided paper, here is a summary focusing on its key contributions, methods, and results:

**Key Contributions:**
This paper introduces **Agentic Context Engineering (ACE)**, a framework designed to enable self-improving language models by treating input contexts as evolving, comprehensive playbooks rather than static or overly concise summaries. ACE addresses two major limitations of existing context adaptation methods: **brevity bias** (the tendency to drop domain-specific details for conciseness) and **context collapse** (the erosion of information during iterative rewriting). The core innovation is a structured, agentic workflow that allows contexts to accumulate and refine detailed strategies over time, making LLM systems more scalable and efficient without requiring weight updates.

**Methods:**
ACE builds upon the agentic architecture of Dynamic Cheatsheet but introduces a modular, three-component workflow:
1.  **Generator:** Produces reasoning trajectories for new queries.
2.  **Reflector:** Critiques these trajectories to extract concrete lessons from successes and failures.
3.  **Curator:** Integrates these distilled insights into the existing context through structured, incremental updates.

The framework employs two key mechanisms to prevent collapse and ensure efficiency:
*   **Incremental Delta Updates:** Instead of fully rewriting the context, ACE makes localized edits by adding compact "delta entries," which preserves past knowledge and reduces computational cost.
*   **Grow-and-Refine:** This process balances the steady expansion of the context with periodic de-duplication to control redundancy and maintain relevance.

**Results:**
The authors evaluated ACE on agent benchmarks (AppWorld) and domain-specific financial reasoning tasks (FiNER, Formula). The results demonstrate significant improvements:
*   ACE consistently outperformed strong baselines like GEPA, MIPROv2, and Dynamic Cheatsheet, achieving average gains of **+10.6% on agent tasks** and **+8.6% on financial benchmarks**.
*   A key finding is that ACE can effectively self-improve **without ground-truth labels**, leveraging only natural execution feedback (e.g., code execution success/failure).
*   On the competitive AppWorld leaderboard, ACE enabled a smaller open-source model (DeepSeek-V3.1) to match the performance of the top-ranked proprietary agent (IBM CUGA powered by GPT-4.1) and even surpass it on the more difficult test split.
*   Crucially, ACE achieved these performance gains with dramatically **lower overhead**, reducing adaptation latency by **86.9%** on average and significantly cutting down on the number of required rollouts and token costs compared to other adaptive methods.

## Critique

Of course. Here is a critique of the paper "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models," focusing on its strengths and weaknesses.

### Strengths

1.  **Novelty and Problem Formulation:** The paper identifies and clearly articulates two significant, under-explored problems in context adaptation for LLMs: **"brevity bias"** (over-compression leading to loss of useful detail) and **"context collapse"** (catastrophic loss of information during iterative rewriting). Proposing that contexts should be "comprehensive, evolving playbooks" rather than concise summaries is a compelling and well-justified paradigm shift, especially with the rise of long-context models.

2.  **Well-Designed Technical Solution:** The ACE framework is a thoughtful evolution of prior work like Dynamic Cheatsheet. Its core innovations are sound and address the stated problems directly:
    *   **Modular Architecture:** Separating the roles of Generator, Reflector, and Curator is a robust design that prevents a single LLM from being overloaded and encourages specialized, higher-quality outputs.
    *   **Incremental Delta Updates:** This is a key technical contribution that directly mitigates context collapse by avoiding monolithic rewrites. It also provides a clear pathway for efficiency gains.
    *   **Grow-and-Refine Mechanism:** This provides a practical method for managing context growth and maintaining relevance, balancing expansion with necessary pruning.

3.  **Significant and Comprehensive Empirical Results:** The experimental evaluation is thorough and convincing. The paper demonstrates strong performance across multiple settings:
    *   **Performance Gains:** Consistent and substantial improvements over strong baselines (GEPA, Dynamic Cheatsheet) on both agent (AppWorld) and domain-specific (financial) benchmarks are reported.
    *   **Efficiency:** The dramatic reductions in latency (up to 91.5%) and cost are a major selling point, making the approach not just more accurate but also more practical for deployment.
    *   **Leaderboard Comparison:** Showing that ACE enables a smaller open-source model to compete with a top-ranked proprietary agent (IBM CUGA) is a powerful result that underscores the method's significance.

4.  **Clarity of Presentation:** The paper is generally well-written and structured. The figures effectively illustrate the core concepts (e.g., the ACE framework diagram, context collapse), and the tables are clear and informative. The discussion section thoughtfully addresses potential concerns about serving costs and connects the work to broader topics like continuous learning.

### Weaknesses

1.  **Limited Analysis of "Brevity Bias":** While the concept of "brevity bias" is introduced as a key motivation, the empirical evidence for it is somewhat anecdotal, relying on a citation to another paper ([16]). The paper would be stronger if it included a quantitative analysis or a clear ablation showing how methods like GEPA actually lose domain-specific details that ACE retains.

2.  **Dependence on Feedback Quality:** The results in Table 2 reveal a significant limitation: ACE's performance can degrade sharply without high-quality feedback signals (e.g., ground-truth labels or clear execution outcomes). This is acknowledged in the analysis but deserves more emphasis. It suggests that ACE's applicability might be limited to tasks where reliable, automatic feedback is readily available, potentially restricting its use in more open-ended or creative domains.

3.  **Ablation Study Could Be Deeper:** The ablation study in Table 3 is good but could be more granular. It bundles the "Reflector" and "multi-epoch" components initially, making it difficult to isolate the individual contribution of the Reflector—which is a central novel component compared to Dynamic Cheatsheet. A more fine-grained ablation separating these elements would provide stronger evidence for the design choices.

4.  **Scalability and Long-Context Management:** While the "grow-and-refine" mechanism is proposed, the paper provides limited discussion on its long-term scalability. How does performance evolve as the context playbook grows to hundreds of thousands of tokens? What are the strategies for retrieval and prioritization within a massive context? A discussion of these potential challenges would strengthen the paper.

### Overall Assessment

This is a **strong paper** with a high degree of novelty and practical impact. It successfully identifies genuine shortcomings in existing context adaptation methods and proposes a well-engineered, efficient framework (ACE) that effectively addresses them. The empirical results are impressive, demonstrating significant performance and efficiency gains across diverse benchmarks. The main weaknesses lie in a slightly under-demonstrated motivation for "brevity bias" and a clearer acknowledgment of the framework's dependency on high-quality feedback. Nonetheless, the contributions are substantial and the work represents a meaningful advance in the development of self-improving, agentic AI systems.

