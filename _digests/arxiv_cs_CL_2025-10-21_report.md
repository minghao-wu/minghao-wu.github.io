---
title: "ArXiv Daily Digest on 2025-10-21"
collection: digests
permalink: /digests/arxiv_cs_CL_2025-10-21_report
date: 2025-10-21
location: "Online"
---

Today's research highlights innovative strategies for enhancing model performance while mitigating critical limitations. A prominent theme is the integration of Large Language Models (LLMs) with specialized frameworks to overcome data and semantic gaps: the **HYDRE (HYbrid Distantly supervised Relation Extraction)** framework combines distantly supervised models with in-context learning to tackle noisy annotations in relation extraction, achieving significant gains in both monolingual and cross-lingual settings. Meanwhile, **CodeRL+** introduces execution semantics alignment into Reinforcement Learning with Verifiable Rewards (RLVR), effectively bridging the gap between textual patterns and functional code correctness. Complementing these advances, a systematic analysis of **catastrophic forgetting** reveals that reinforcement learning (RL), due to its use of on-policy data, consistently preserves prior knowledge better than supervised fine-tuning (SFT) during post-training, offering practical guidelines for continual adaptation.

## TL;DR

Total papers: 53 , Selected papers: 3

Here's a TL;DR summary of the key themes and insights from these papers:

**Main Theme: Enhancing Language Models through Hybrid Approaches and On-Policy Learning**

These papers explore innovative methods to improve language model capabilities while addressing fundamental limitations:

**HYDRE** (https://arxiv.org/abs/2510.18344v1) introduces a hybrid framework that combines distantly supervised relation extraction models with LLMs via in-context learning. The key insight is using a trained DSRE model to identify candidate relations, then retrieving reliable exemplars for LLM prompting. This approach achieves up to 20 F1 point gains in English and 17 F1 points on average for low-resource Indic languages, demonstrating the power of combining traditional models with LLM reasoning.

**Retaining by Doing** (https://arxiv.org/abs/2510.18874v1) reveals the surprising finding that reinforcement learning causes significantly less catastrophic forgetting than supervised fine-tuning during language model post-training. The critical factor is RL's use of on-policy data, which enables models to preserve existing knowledge while learning new tasks. This provides practical guidelines for model adaptation where knowledge retention is crucial.

**CodeRL+** (https://arxiv.org/abs/2510.18471v1) addresses the semantic gap between textual code patterns and execution semantics by integrating variable-level execution trajectory alignment into RL training. This provides dense learning signals beyond binary test outcomes, achieving 4.6% average improvement in code generation and strong generalization to reasoning tasks.

**Common Insights:**
- Hybrid approaches combining specialized models with LLMs yield substantial performance gains
- On-policy data generation is crucial for mitigating forgetting during model adaptation
- Bridging semantic gaps between training objectives and real-world functionality requires novel alignment strategies
- These methods show consistent improvements across diverse tasks, models, and languages

---

# Combining Distantly Supervised Models with In Context Learning for Monolingual and Cross-Lingual Relation Extraction

Authors: Vipul Rathore, Malik Hammad Faisal, Parag Singla, Mausam

Keywords: Distantly Supervised Relation Extraction, In-Context Learning, Cross-Lingual Transfer, Low-Resource Languages, Large Language Models, Multilingual NLP, Hybrid Framework

Comments: None

Paper link: [http://arxiv.org/abs/2510.18344v1](http://arxiv.org/abs/2510.18344v1)

## Abstract

Distantly Supervised Relation Extraction (DSRE) remains a long-standing challenge in NLP, where models must learn from noisy bag-level annotations while making sentence-level predictions. While existing state-of-the-art (SoTA) DSRE models rely on task-specific training, their integration with in-context learning (ICL) using large language models (LLMs) remains underexplored. A key challenge is that the LLM may not learn relation semantics correctly, due to noisy annotation.   In response, we propose HYDRE -- HYbrid Distantly Supervised Relation Extraction framework. It first uses a trained DSRE model to identify the top-k candidate relations for a given test sentence, then uses a novel dynamic exemplar retrieval strategy that extracts reliable, sentence-level exemplars from training data, which are then provided in LLM prompt for outputting the final relation(s).   We further extend HYDRE to cross-lingual settings for RE in low-resource languages. Using available English DSRE training data, we evaluate all methods on English as well as a newly curated benchmark covering four diverse low-resource Indic languages -- Oriya, Santali, Manipuri, and Tulu. HYDRE achieves up to 20 F1 point gains in English and, on average, 17 F1 points on Indic languages over prior SoTA DSRE models. Detailed ablations exhibit HYDRE's efficacy compared to other prompting strategies.

## Summary

Based on the provided paper, here is a summary focusing on its key contributions, methods, and results:

**Key Contributions:**
This paper introduces **Hydre (HYbrid Distantly supervised Relation Extraction)**, a novel framework that integrates distantly supervised relation extraction (DSRE) models with large language models (LLMs) through in-context learning (ICL). A significant contribution is the extension of this framework to cross-lingual settings, supported by a newly curated benchmark for four low-resource Indic languages: Oriya, Santali, Manipuri, and Tulu. This addresses a major gap in multilingual DSRE research.

**Methods:**
Hydre employs a three-stage, hybrid pipeline to overcome the noise in bag-level DSRE annotations and leverage LLMs' reasoning capabilities for sentence-level inference:
1.  **Candidate Relation Selection:** A pre-trained DSRE model (e.g., PARE) identifies a high-recall set of top-k candidate relations for a given test sentence.
2.  **Bag Selection:** For each candidate relation, the most relevant bag from the training data is retrieved using a joint scoring function that combines the DSRE model's confidence and semantic similarity to the query.
3.  **Sentence Selection:** From each selected bag, the most representative sentence is extracted based on its coverage of the bag's relations and aggregate model confidence. These clean, sentence-level exemplars are then used in a prompt to guide an LLM in disambiguating and selecting the final relation(s) for the test query.
The method is evaluated in monolingual (English) and three cross-lingual settings (English-only, Translate-train, Translate-test), demonstrating its adaptability.

**Results:**
Hydre achieves state-of-the-art performance, significantly outperforming prior DSRE models and naive LLM prompting strategies:
*   **Monolingual (English):** Hydre achieves up to **63 micro-F1** on the NYT-10m dataset, a gain of over 20 F1 points compared to supervised DSRE baselines like PARE (42 F1) and CIL (43 F1).
*   **Cross-lingual (Indic Languages):** Hydre shows substantial improvements in low-resource settings, achieving an average gain of **17 F1 points** over prior SoTA models. For instance, in the Translate-train setting, it reaches **47 micro-F1** with a fine-tuned Llama model, compared to 30 F1 for PARE-X.
*   **Ablation studies** confirm the importance of each component in Hydre's pipeline, showing that removing semantic similarity or sentence selection can degrade performance by up to 7 F1 points. The method also proves more effective than other exemplar selection strategies like similarity-based or diversity-aware retrieval.

## Critique

Of course. Here is a critique of the paper "Combining Distantly Supervised Models with In Context Learning for Monolingual and Cross-Lingual Relation Extraction."

### Summary of Strengths

1.  **Novel and Well-Motivated Hybrid Approach:** The core idea of Hydre—using a high-recall DSRE model (PARE) to generate candidate relations and then leveraging an LLM as a "judge" with carefully selected exemplars—is both novel and elegant. It effectively addresses the key weakness of DSRE (sentence-level inference from bag-level, noisy training) by combining the strengths of two different paradigms. The motivation is clear: DSRE models have high recall but struggle with fine-grained disambiguation, while LLMs have strong reasoning capabilities but falter with noisy data.

2.  **Comprehensive and Significant Evaluation:** The paper provides an extensive empirical evaluation across multiple dimensions:
    *   **Models:** It tests a wide range of LLMs, from open-source (Llama, Qwen) to proprietary (GPT-4o), and compares against strong supervised DSRE baselines.
    *   **Settings:** It thoroughly explores monolingual (English) and three distinct cross-lingual transfer settings (English-only, Translate-train, Translate-test), demonstrating the framework's versatility.
    *   **Ablations:** The ablation studies are systematic and insightful, clearly demonstrating the contribution of each component (candidate selection, bag selection, sentence selection) and the interplay between semantic similarity and model confidence in different contexts.

3.  **Valuable Resource for the Community:** The creation and release of a manually verified, gold-standard benchmark for four low-resource Indic languages (Oriya, Santali, Manipuri, Tulu) is a significant contribution. This addresses a clear gap in the field and provides a valuable testbed for future research in multilingual and low-resource NLP.

4.  **Strong and Consistent Results:** The performance gains are substantial and consistent. Achieving up to 20 F1 point improvements over prior state-of-the-art models in English and an average of 17 points on Indic languages is a remarkable result. The fact that Hydre provides significant boosts even to powerful models like GPT-4o, and even larger gains for smaller models, underscores its effectiveness.

5.  **Clear Presentation and Analysis:** The paper is generally well-structured. The three-stage Hydre algorithm is explained clearly with the aid of a figure. The results are presented comprehensively, and the subsequent analysis (qualitative, error, sensitivity) provides a deep dive into the model's behavior, strengths, and failure modes.

### Summary of Weaknesses

1.  **Computational Cost and Latency:** The paper explicitly acknowledges this as a limitation, and it is a significant one. The multi-stage retrieval process (scoring bags, then sentences) against a large training corpus, combined with the cost of LLM inference (especially with long, exemplar-filled prompts), makes Hydre computationally expensive and likely slow compared to a single forward pass of a fine-tuned DSRE model. This could hinder its use in real-time applications.

2.  **Limited Domain and Language Scope:** As noted in the limitations, the evaluation is confined to the general-domain NYT and Wiki datasets. Its performance on specialized domains (e.g., biomedical, financial) with complex, domain-specific relations remains unknown. Furthermore, while the four Indic languages are a good start, the claim of robustness in "low-resource" settings would be stronger if validated across a wider array of language families.

3.  **Error Analysis Could Be Deeper:** While an error analysis is provided, it is somewhat high-level. A more quantitative breakdown of error types (e.g., percentage of errors due to position bias, multi-label recall failure, or specific relation confusion) would provide a clearer picture. The proposed solution for bag-level evaluation ("reduced bag") is interesting but feels preliminary and could benefit from a more formalized algorithm and evaluation.

4.  **Positional Bias in Prompting:** The error analysis identifies a "position bias" where the LLM favors the first candidate relation. This is a known issue in LLM prompting. While identifying it is good, the paper does not explore or propose methods to mitigate it (e.g., prompt randomization, ensembling), which would be a valuable addition.

5.  **Clarity on "w/o PARE confidence" in Cross-lingual Setting:** In the cross-lingual ablations (Table 4), the "only sem. sim." variant performs nearly as well as the full Hydre for some models. The explanation for why semantic retrieval hurts performance is relegated to the appendix. This critical finding—that model confidence is more important than semantic similarity for low-resource languages—deserves a more prominent discussion in the main text to fully clarify the cross-lingual dynamics.

### Overall Assessment

This is a strong paper that makes a clear and valuable contribution to the fields of Relation Extraction and Multilingual NLP. The proposed Hydre framework is novel, well-executed, and demonstrates substantial empirical improvements over existing methods. The strengths far outweigh the weaknesses. The primary limitations regarding cost and scope are openly acknowledged and represent clear and compelling directions for future work. The paper is likely to influence subsequent research by demonstrating a powerful new paradigm for combining traditional fine-tuned models with the reasoning capabilities of large language models.

---

# Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting

Authors: Howard Chen, Noam Razin, Karthik Narasimhan, Danqi Chen

Keywords: catastrophic forgetting, language model post-training, supervised fine-tuning, reinforcement learning, on-policy data, KL divergence

Comments: None

Paper link: [http://arxiv.org/abs/2510.18874v1](http://arxiv.org/abs/2510.18874v1)

## Abstract

Adapting language models (LMs) to new tasks via post-training carries the risk of degrading existing capabilities -- a phenomenon classically known as catastrophic forgetting. In this paper, toward identifying guidelines for mitigating this phenomenon, we systematically compare the forgetting patterns of two widely adopted post-training methods: supervised fine-tuning (SFT) and reinforcement learning (RL). Our experiments reveal a consistent trend across LM families (Llama, Qwen) and tasks (instruction following, general knowledge, and arithmetic reasoning): RL leads to less forgetting than SFT while achieving comparable or higher target task performance. To investigate the cause for this difference, we consider a simplified setting in which the LM is modeled as a mixture of two distributions, one corresponding to prior knowledge and the other to the target task. We identify that the mode-seeking nature of RL, which stems from its use of on-policy data, enables keeping prior knowledge intact when learning the target task. We then verify this insight by demonstrating that the use on-policy data underlies the robustness of RL to forgetting in practical settings, as opposed to other algorithmic choices such as the KL regularization or advantage estimation. Lastly, as a practical implication, our results highlight the potential of mitigating forgetting using approximately on-policy data, which can be substantially more efficient to obtain than fully on-policy data.

## Summary

This paper "Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting" systematically compares catastrophic forgetting in supervised fine-tuning (SFT) and reinforcement learning (RL) for language model post-training. The key finding is that RL consistently exhibits substantially less forgetting than SFT while achieving comparable or better target task performance across instruction following, general knowledge, and reasoning tasks, using various model families (Llama, Qwen) and scales (1B-8B parameters).

The authors provide theoretical intuition through a simplified mixture-of-Gaussians analysis, showing that while mode-covering forward KL (SFT) should intuitively preserve old knowledge better, in practical multi-modal LM settings, the mode-seeking reverse KL (RL) actually causes less forgetting. They demonstrate that RL's robustness stems primarily from its use of on-policy data rather than other algorithmic factors like KL regularization or advantage estimation. Importantly, the paper shows that even approximately on-policy data (generated at the start of each epoch in Iterative-SFT) can substantially reduce forgetting compared to fully off-policy SFT, suggesting more efficient forgetting mitigation strategies.

This work provides practical guidelines for LM post-training, highlighting that on-policy data generation is crucial for preserving existing capabilities while adapting to new tasks, with implications for continual learning and agent development where preserving knowledge during adaptation is essential.

## Critique

Of course. Here is a critique of the paper "Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting," focusing on its strengths, weaknesses, and overall contribution.

### Summary

This paper provides a systematic empirical and theoretical investigation into catastrophic forgetting during language model (LM) post-training. The central finding is that Reinforcement Learning (RL) is significantly more robust to forgetting than Supervised Fine-Tuning (SFT), and the authors identify the use of **on-policy data** as the primary mechanism for this robustness, rather than other algorithmic components like KL regularization or advantage estimation.

---

### Strengths

1.  **Compelling and Counter-Intuitive Core Finding:** The central result—that the "mode-seeking" RL forgets less than the "mode-covering" SFT—is surprising and challenges conventional wisdom. This makes the paper's contribution immediately interesting and significant. It directly addresses a critical pain point in LM adaptation ("alignment tax") with a clear, evidence-based finding.

2.  **Rigorous and Comprehensive Empirical Evaluation:** The experimental design is thorough. The authors:
    *   Test across multiple **model families** (Llama, Qwen) and **scales** (1B to 8B).
    *   Evaluate on diverse **task types** (instruction following, knowledge, reasoning, safety).
    *   Compare multiple **baselines** (SFT, Self-SFT, RL/GRPO, REINFORCE).
    This multi-faceted approach makes the core claim—that RL is more robust—very convincing and generalizable.

3.  **Effective Use of Ablation Studies:** The paper excellently isolates the cause of the phenomenon. By systematically removing components of RL (KL regularization, advantage estimation), they provide strong evidence that the key factor is indeed the **on-policy nature of the data**, not the other algorithmic bells and whistles. This is a crucial contribution.

4.  **Insightful Theoretical Intuition:** The simplified mixture-of-Gaussians simulation in Section 3 is a major strength. It elegantly bridges the gap between the counter-intuitive empirical result and a theoretical understanding. The distinction between uni-modal and multi-modal initial policies provides a clear, intuitive explanation for *why* the conventional wisdom fails in the context of fine-tuning large, multi-capability LMs. Figure 1 is an excellent visual summary of this concept.

5.  **Practical Implications and "Approximately On-Policy" Data:** The investigation into Iterative-SFT is a valuable practical contribution. It shows that one doesn't necessarily need the full compute overhead of online RL; periodically refreshing the dataset with on-policy data can substantially mitigate forgetting. This provides a concrete, more efficient guideline for practitioners.

### Weaknesses

1.  **Limited Scale and Scope:** While the evaluation across model families is a strength, the largest model tested is 8B parameters. Forgetting dynamics in today's state-of-the-art models (e.g., 70B+ parameters) could be different. Furthermore, the tasks, while diverse, are still limited. Exploring more complex, open-ended generation tasks would strengthen the findings.

2.  **Theoretical Grounding is Intuitive but Not Formal:** The analysis in Section 3 is based on a simulation with simple, low-dimensional distributions. While highly effective for building intuition, it is not a formal proof or a theoretical analysis of the dynamics in high-dimensional parameter spaces of transformers. The paper identifies the "what" (on-policy data) and provides a compelling "why" (multi-modal mode-seeking), but a more rigorous theoretical foundation would elevate the work further.

3.  **Handling of Concurrent Work:** The authors note concurrent work (Lai et al., Shenfeld et al.) that made similar observations. While they do differentiate their work (e.g., by arguing against Lai et al.'s advantage estimation hypothesis), the paper would be stronger with a more detailed comparative discussion in the main text, perhaps in a dedicated section, to more clearly stake out its unique contribution within this emerging research thread.

4.  **Clarity of the "Approximately On-Policy" Finding:** The result for Iterative-SFT is promising, but the explanation could be clearer. How "close" does the policy need to be to the current one for the data to be "approximately on-policy"? Is one epoch the right frequency, or is this task-dependent? A more nuanced discussion of the trade-offs (compute vs. forgetting reduction) would be helpful.

### Overall Assessment

**Novelty:** High. The core finding that RL forgets less than SFT is counter-intuitive and significant. The systematic identification of on-policy data as the causal mechanism, supported by strong ablations and a clear theoretical intuition, constitutes a novel and important contribution to the field.

**Significance:** Very High. Catastrophic forgetting is a fundamental challenge in machine learning and a major practical obstacle in adapting large language models. This paper provides a clear, empirically-backed guideline (favor on-policy or approximately on-policy learning) that can immediately influence how researchers and practitioners approach LM post-training and continual learning.

**Clarity of Presentation:** Excellent. The paper is well-structured, the figures are informative and central to the narrative, and the writing is clear and direct. The progression from empirical discovery -> theoretical intuition -> causal ablation -> practical implication is logical and easy to follow.

**Conclusion:** This is a high-quality paper that makes a substantial contribution. It combines a surprising empirical discovery with insightful analysis to provide both a deeper understanding of forgetting dynamics and practical guidance for mitigating it. The weaknesses are minor and primarily point to fruitful directions for future work.

---

# CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment

Authors: Xue Jiang, Yihong Dong, Mengyang Liu, Hongyi Deng, Tian Wang, Yongding Tao, Rongyu Cao, Binhua Li, Zhi Jin, Wenpin Jiao, Fei Huang, Yongbin Li, Ge Li

Keywords: Code Generation, Reinforcement Learning, Execution Semantics, Program Synthesis, Code Correctness, RLVR, Policy Optimization

Comments: None

Paper link: [http://arxiv.org/abs/2510.18471v1](http://arxiv.org/abs/2510.18471v1)

## Abstract

While Large Language Models (LLMs) excel at code generation by learning from vast code corpora, a fundamental semantic gap remains between their training on textual patterns and the goal of functional correctness, which is governed by formal execution semantics. Reinforcement Learning with Verifiable Rewards (RLVR) approaches attempt to bridge this gap using outcome rewards from executing test cases. However, solely relying on binary pass/fail signals is inefficient for establishing a well-aligned connection between the textual representation of code and its execution semantics, especially for subtle logical errors within the code. In this paper, we propose CodeRL+, a novel approach that integrates execution semantics alignment into the RLVR training pipeline for code generation. CodeRL+ enables the model to infer variable-level execution trajectory, providing a direct learning signal of execution semantics. CodeRL+ can construct execution semantics alignment directly using existing on-policy rollouts and integrates seamlessly with various RL algorithms. Extensive experiments demonstrate that CodeRL+ outperforms post-training baselines (including RLVR and Distillation), achieving a 4.6% average relative improvement in pass@1. CodeRL+ generalizes effectively to other coding tasks, yielding 15.5% and 4.4% higher accuracy on code-reasoning and test-output-generation benchmarks, respectively. CodeRL+ shows strong applicability across diverse RL algorithms and LLMs. Furthermore, probe analyses provide compelling evidence that CodeRL+ strengthens the alignment between code's textual representations and its underlying execution semantics.

## Summary

Here is a summary of the paper "CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment":

**Key Problem & Contribution:** The paper addresses a fundamental limitation in current code generation with Large Language Models (LLMs): the semantic gap between learning from textual patterns during pre-training and the goal of producing functionally correct code, which is defined by formal execution semantics. While Reinforcement Learning with Verifiable Rewards (RLVR) uses test case outcomes (pass/fail) as rewards, it provides only sparse, binary signals. The key contribution is **CodeRL+**, a novel method that integrates fine-grained **execution semantics alignment** directly into the RL training pipeline to bridge this gap more effectively.

**Core Methodology:** CodeRL+ enhances standard RLVR (e.g., GRPO) with a dual-objective optimization framework. Alongside the primary code generation task, it introduces a parallel **execution semantics alignment** task. This task requires the model to infer the final values of all variables in a program's execution trace, providing a dense, direct learning signal about the code's runtime behavior. Crucially, the data for this alignment task is generated on-the-fly by repurposing *failed code samples* from the model's own rollouts, creating a dynamic and evolving training curriculum that requires no external data sources.

**Key Results:**
- **State-of-the-Art Performance:** CodeRL+ outperforms strong baselines (including other RLVR methods and distillation-based models like OlympicCoder and CODEI/O) on major code generation benchmarks (HumanEval, LeetCode, LiveCodeBench), achieving an average relative improvement of 4.6% in pass@1.
- **Strong Generalization:** It demonstrates significant gains on code-related tasks beyond pure generation, achieving 15.5% and 4.4% higher accuracy on code-reasoning and test-output-generation benchmarks, respectively.
- **Broad Applicability:** The method shows consistent improvements across different LLM families (LLaMA, Qwen) and sizes (1.5B to 8B parameters), and is compatible with various RL algorithms (GRPO, PPO, REINFORCE++), boosting PPO's performance by +7.4% on average.
- **Proven Alignment:** Probing analysis provides empirical evidence that CodeRL+ strengthens the alignment between the model's internal textual representations and the actual execution semantics of code, a core goal of the method.

## Critique

Of course. Here is a critique of the paper "CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment," covering its strengths, weaknesses, and overall assessment.

### Strengths

1.  **Novel and Well-Motivated Core Idea:** The paper addresses a fundamental and widely recognized problem in LLM-based code generation: the "semantic gap" between learning textual patterns and understanding functional, execution-based correctness. The proposal to integrate "execution semantics alignment" directly into the RL training loop is a novel and compelling approach. Moving beyond sparse, binary (pass/fail) rewards to dense, variable-level supervision is a significant conceptual advance.

2.  **Strong and Comprehensive Empirical Evaluation:** The experimental section is a major strength. The authors demonstrate the effectiveness of CodeRL+ across:
    *   **Multiple Tasks:** Code generation, code reasoning, and test output generation.
    *   **Multiple Benchmarks:** HumanEval, LeetCode, LiveCodeBench.
    *   **Multiple Models:** Different families (LLaMA, Qwen) and sizes (1.5B, 7B, 8B).
    *   **Multiple RL Algorithms:** GRPO, PPO, REINFORCE++.
    This thoroughness makes the claims of generalizability highly convincing.

3.  **Effective and "Free" Data Strategy:** A clever aspect of the methodology is the on-the-fly construction of the alignment dataset from the model's own **failed rollouts**. This is computationally efficient, requires no external data sources, and ensures the training examples are directly relevant to the model's current weaknesses, creating a self-improving loop.

4.  **Insightful Analysis:** The paper goes beyond just reporting performance numbers. The ablation studies effectively validate key design choices (use of failed rollouts, on-policy data, variable-level vs. IO-level supervision). The training dynamics plots and the probing analysis provide valuable insights into *why* the method works, showing that the model's internal representations become more aligned with execution semantics.

### Weaknesses

1.  **Clarity and Presentation of the Core Methodology (Section 3.2):** While the high-level idea is clear, the formalization and description of the "execution semantics alignment" task could be more accessible. The definition of ℱ̂_p(x) and its approximation, while precise, is dense. A more concrete, step-by-step example in the main text (similar to the one in Figure 1a but for the training process) would greatly improve clarity. The connection between the generated code `p` and the alignment prompt `q'` could be explained more explicitly.

2.  **Limited Discussion of Computational Overhead:** The method introduces additional forward passes for the alignment tasks and requires executing failed programs to extract variable traces. While the results are impressive, a discussion of the associated computational cost compared to standard GRPO—even a rough estimate of the increase in training time or FLOPs—would be important for practitioners considering adoption.

3.  **Superficial Treatment of Limitations:** The limitations section in the appendix is very brief. It mentions the challenge with non-deterministic code and programs with side effects but does not explore other potential pitfalls. For instance:
    *   How does the method scale to extremely long programs with many variables?
    *   What is the performance on programs where the "last definition point" of a variable is ambiguous or complex due to intricate control flow?
    *   Could the focus on variable final values miss important intermediate state errors in certain types of algorithms? A more thorough discussion would strengthen the paper.

### Overall Assessment

**Novelty:** High. The core idea of bridging the textual-execution semantic gap through direct, variable-level reinforcement learning is a distinct and valuable contribution to the field.

**Significance of Results:** Very High. The consistent and substantial improvements across a wide range of models, tasks, and RL backbones are compelling. Achieving state-of-the-art results on competitive benchmarks while also significantly boosting performance on related tasks like code reasoning demonstrates the power and generalizability of the approach.

**Clarity of Presentation:** Good, but with room for improvement. The paper is well-structured, and the figures are effective. The primary weakness is the somewhat technical and dense presentation of the core algorithm in Section 3.2, which could be made more intuitive.

In conclusion, this is a strong paper with a novel, well-motivated idea that is backed by extensive and convincing experiments. The weaknesses are primarily related to the clarity of the methodological details and a more in-depth discussion of costs and limitations, but they do not detract significantly from the paper's substantial contributions.

