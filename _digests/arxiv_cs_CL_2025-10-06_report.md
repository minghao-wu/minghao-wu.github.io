---
title: "ArXiv Daily Digest on 2025-10-06"
collection: digests
type: "ArXiv daily digest"
permalink: /digests/arxiv_cs_CL_2025-10-06_report
date: 2025-10-06
location: "Online"
---

Today's research landscape showcases significant advancements in multi-agent collaboration and model orchestration, with several papers exploring how AI systems can work together more effectively. A key theme is the development of specialized frameworks for orchestrating Small Language Models (SLMs), where SLM-MUX demonstrates that carefully selected ensembles can outperform much larger monolithic models on complex reasoning tasks. In parallel, Multi-Agent Tool-Integrated Policy Optimization (MATPO) enables distinct planner and worker roles to be trained within a single Large Language Model (LLM), achieving substantial performance gains while avoiding the overhead of deploying multiple separate models. These developments are complemented by novel evaluation benchmarks like LLM-Hanabi, which provides automated assessment of Theory-of-Mind (ToM) capabilities in dynamic collaborative settings, revealing that first-order ToM (interpreting others' intent) correlates more strongly with success than complex recursive reasoning.

## TL;DR

Here's a TL;DR summary of the key themes and insights from these papers:

**Multi-Agent Collaboration & Reasoning:** Several papers explore multi-agent systems where language models collaborate. LLM-Hanabi introduces a benchmark showing that first-order Theory-of-Mind (interpreting others' intent) is more critical for collaboration success than complex recursive reasoning. MATPO enables training planner-worker agents within a single LLM, achieving 18%+ performance gains while avoiding multi-model overhead.

**Efficient Model Orchestration:** SLM-MUX demonstrates that orchestrating small language models through confidence-based selection (rather than discussion) can match or outperform much larger models, providing a cost-effective "multi-core" alternative to scaling monolithic models.

**Alignment & Optimization:** BVPO addresses alignment challenges in Large Reasoning Models by optimizing the bias-variance trade-off in preference optimization, reducing gradient variance from stochastic trace sampling and improving both alignment and reasoning performance.

**Multilingual & Architectural Insights:** Analysis of Mixture-of-Experts models reveals language-specific routing in early/late layers with universal expert sharing in middle layers, where routing alignment with English correlates strongly with multilingual performance.

**Information Theory & Efficiency:** New methods for Partial Information Decomposition using normalizing flows enable efficient quantification of redundant, unique, and synergistic information in multimodal data, with applications to model selection and dataset analysis.

**Key Insight:** Across these papers, we see a trend toward specialized, efficient systems (multi-agent, SLM orchestration, MoE routing) that outperform monolithic approaches through smart coordination and optimization, rather than simply scaling model size.

---

# From Noisy Traces to Stable Gradients: Bias-Variance Optimized Preference Optimization for Aligning Large Reasoning Models

Authors: Mingkang Zhu, Xi Chen, Bei Yu, Hengshuang Zhao, Jiaya Jia

Keywords: Multi-Agent Collaboration, Theory-of-Mind, Rationale Inference, Imperfect Information Games, Large Language Model Evaluation, Cooperative Gameplay

Comments: None

Paper link: [http://arxiv.org/abs/2510.05095v1](http://arxiv.org/abs/2510.05095v1)

## Abstract

Large reasoning models (LRMs) generate intermediate reasoning traces before producing final answers, yielding strong gains on multi-step and mathematical tasks. Yet aligning LRMs with human preferences, a crucial prerequisite for model deployment, remains underexplored. The statistically correct objective for preference alignment requires marginalizing over reasoning traces, but this computation is intractable in practice. A common workaround optimizes a single sampled trajectory, which introduces substantial gradient variance from stochastic trace sampling. To address this challenge, we frame preference optimization for LRMs through the lens of the bias--variance trade-off and propose Bias--Variance Optimized Preference Optimization (BVPO), a simple, drop-in method that mixes two gradient estimators: a high-variance trace-based estimator and a low-variance empty-trace estimator obtained by disabling reasoning trace generation. Our theory shows that BVPO strictly reduces trace-induced variance for any nontrivial mixture, provides a closed-form choice of the mixing weight that minimizes mean-squared error relative to the true marginal gradient, and under standard smoothness and step-size conditions, tightens classical convergence bounds for stochastic gradient descent. Empirically, BVPO improves alignment over the best baseline by up to 7.8 points on AlpacaEval~2 and 6.8 points on Arena-Hard. Despite being trained only on general conversational data, BVPO also boosts reasoning performance for base models by up to 4.0 points on the average of six math reasoning benchmarks. These results identify variance from trace sampling as a key bottleneck and demonstrate that directly optimizing the bias--variance trade-off yields more stable training and stronger overall performance.

## Summary

This paper introduces **LLM-Hanabi**, a novel benchmark designed to evaluate **Theory-of-Mind (ToM)** and **rationale inference** in large language models (LLMs) within a dynamic, multi-agent collaboration setting. The benchmark is built upon the cooperative card game *Hanabi*, where players have imperfect information and must rely on interpreting sparse linguistic hints to succeed. The key contribution is an automated evaluation framework that measures both game performance and ToM proficiency, addressing a gap in existing benchmarks that are often static and fail to capture the interactive nature of real-world collaboration.

The methodology involves using LLM-driven agents to play Hanabi in a 5-player configuration, with the game state translated into natural language. During gameplay, agents generate structured reasoning statements: *Rationale* (the hinter's intent), *First-Order ToM* (the recipient's interpretation of the intent), and *Second-Order ToM* (the hinter's prediction of the recipient's interpretation). An LLM-as-a-judge then evaluates these statements post-game to produce quantitative ToM scores, providing a scalable assessment of collaborative reasoning.

The results, evaluated across a diverse set of LLMs and large reasoning models (LRMs), reveal two main findings: (1) **LRMs significantly outperform standard LLMs** in both game performance and ToM capabilities, with models like Deepseek-R1 and GPT-4.1 achieving the highest scores; (2) There is a **strong positive correlation between ToM proficiency and game success**, with first-order ToM (interpreting others' intent) showing a stronger correlation (r=0.76) with performance than second-order ToM (predicting others' interpretations; r=0.58). This indicates that accurate inference of a partner's rationale is more critical for effective collaboration than higher-order reasoning. The benchmark provides a valuable tool for future research aimed at enhancing the collaborative capabilities of AI systems.

## Critique

Of course. Here is a critique of the paper "LLM-Hanabi: Evaluating Multi-Agent Gameplays with Theory-of-Mind and Rationale Inference in Imperfect Information Collaboration Game."

### Overall Summary

This is a strong, well-executed paper that makes a clear and valuable contribution to the field of multi-agent AI evaluation. It presents a novel benchmark and provides insightful empirical results that challenge a common assumption about the hierarchy of Theory-of-Mind (ToM) reasoning.

---

### Strengths

1.  **Novelty and Significance of the Benchmark:**
    *   **Fills a Clear Gap:** The authors correctly identify a limitation in existing ToM benchmarks, which are often static and text-based (e.g., story QA). LLM-Hanabi addresses this by providing a dynamic, interactive environment where reasoning must happen in real-time under uncertainty.
    *   **Ideal Testbed:** The choice of Hanabi is excellent. Its cooperative nature, imperfect information, and reliance on sparse linguistic hints perfectly isolate the core challenges of collaborative reasoning and rationale inference, avoiding the confounding factors of deception found in adversarial games.

2.  **Well-Designed Evaluation Framework:**
    *   The automated "LLM-as-a-judge" system for scoring first-order and second-order ToM is a clever and scalable solution. By extracting structured rationales during gameplay, they create a traceable and quantifiable metric for a typically qualitative concept.
    *   The evaluation is comprehensive, testing a wide range of models (both LLMs and the newer LRMs) and reporting multiple relevant metrics (game score, rounds, and both orders of ToM).

3.  **Significant and Actionable Results:**
    *   The key finding—that **first-order ToM is a stronger predictor of collaborative success than second-order ToM**—is both surprising and highly significant. It provides a crucial, data-driven insight for the community: for practical collaboration, accurately interpreting your partner's intent is more critical than engaging in complex recursive reasoning about what they think you think.
    *   The clear performance gap between standard LLMs and Large Reasoning Models (LRMs) is a valuable data point that underscores the importance of specialized reasoning architectures for complex, multi-step tasks.

4.  **Clarity of Presentation:**
    *   The paper is exceptionally well-structured and easy to follow. The contributions are listed clearly in the introduction.
    *   Figures 1 and 2 effectively illustrate the evaluation pipeline and the core correlation finding.
    *   Table 1 is comprehensive and allows for easy comparison across models.

---

### Weaknesses and Potential Improvements

1.  **Limited Model Analysis and "Why" Behind Performance:**
    *   The paper thoroughly documents *what* models perform well but offers less insight into *why*. A deeper analysis into the failure modes (e.g., common types of misinterpretations in first-order ToM, or what makes second-order ToM so difficult) would greatly enhance the paper.
    *   For instance, are the errors in second-order ToM due to a fundamental limitation in recursive reasoning, or are they a consequence of compounding errors from imperfect first-order understanding?

2.  **Dependence on LLM-as-Judge:**
    *   While the authors appropriately list this as a limitation, it remains a significant methodological concern. The ToM scores are only as reliable as the judge model's own ToM capabilities. A small-scale human evaluation to validate the judge's scoring would have strengthened the results considerably.

3.  **Clarity on "Rationale Inference" vs. Standard ToM:**
    *   The paper introduces the term "rationale inference" alongside ToM. While the concept is clear in context, the paper could more explicitly define how it relates to or differs from the established definitions of first and second-order ToM. Is it a refinement, a subset, or a synonymous term? A more precise theoretical framing would be beneficial.

4.  **Potential for Overstatement:**
    *   The conclusion that "prioritizing first-order ToM is a promising direction" is well-supported. However, one should be cautious about generalizing this too far. Hanabi is a specific type of collaboration; in other scenarios (e.g., negotiation, teaching), higher-order ToM might be far more critical. The paper could more strongly caveat its findings to this specific collaborative paradigm.

### Conclusion

This is a high-quality paper that makes a substantive contribution. Its primary strength lies in its elegant benchmark design and its empirically-grounded, counter-intuitive finding regarding the relative importance of first-order ToM. The weaknesses are primarily opportunities for deeper analysis rather than fundamental flaws. The LLM-Hanabi benchmark is likely to be adopted by the community, and the paper's core insight will influence how researchers think about and build collaborative AI agents.

---

# Slm-mux: Orchestrating small language models for reasoning

Authors: Chenyu Wang, Zishen Wan, Hao Kang, Emma Chen, Zhiqiang Xie, Tushar Krishna, Vijay Janapa Reddi, Yilun Du

Keywords: Large Reasoning Models, Preference Optimization, Bias-Variance Trade-off, Gradient Variance, Direct Preference Optimization, Trace Sampling, Alignment, Mathematical Reasoning

Comments: None

Paper link: [http://arxiv.org/abs/2510.05077v1](http://arxiv.org/abs/2510.05077v1)

## Abstract

With the rapid development of language models, the number of small language models (SLMs) has grown significantly. Although they do not achieve state-of-the-art accuracy, they are more efficient and often excel at specific tasks. This raises a natural question: can multiple SLMs be orchestrated into a system where each contributes effectively, achieving higher accuracy than any individual model? Existing orchestration methods have primarily targeted frontier models (e.g., GPT-4) and perform suboptimally when applied to SLMs. To address this gap, we propose a three-stage approach for orchestrating SLMs. First, we introduce SLM-MUX, a multi-model architecture that effectively coordinates multiple SLMs. Building on this, we develop two optimization strategies: (i) a model selection search that identifies the most complementary SLMs from a given pool, and (ii) test-time scaling tailored to SLM-MUX. Our approach delivers strong results: Compared to existing orchestration methods, our approach achieves up to 13.4% improvement on MATH, 8.8% on GPQA, and 7.0% on GSM8K. With just two SLMS, SLM-MUX outperforms Qwen 2.5 72B on GPQA and GSM8K, and matches its performance on MATH. We further provide theoretical analyses to substantiate the advantages of our method. In summary, we demonstrate that SLMs can be effectively orchestrated into more accurate and efficient systems through the proposed approach.

## Summary

This paper addresses the challenge of aligning Large Reasoning Models (LRMs) with human preferences, identifying a key bottleneck: the high gradient variance caused by stochastic sampling of reasoning traces during preference optimization. While the statistically correct approach requires marginalizing over all possible reasoning traces, this is computationally intractable in practice. Current methods like DPO instead optimize using single sampled traces, which introduces substantial noise and instability during training.

The authors propose Bias–Variance Optimized Preference Optimization (BVPO), a simple yet effective method that combines two gradient estimators: a high-variance trace-based estimator and a low-variance empty-trace estimator obtained by disabling reasoning trace generation. BVPO forms a convex combination of these estimators, explicitly optimizing the bias-variance trade-off by minimizing the Mean Squared Error relative to the ideal marginal gradient. Theoretically, the authors prove that BVPO strictly reduces trace-induced variance, provides a closed-form optimal mixing weight, and tightens SGD convergence bounds under standard conditions.

Empirical results demonstrate BVPO's effectiveness across three LRMs. On alignment benchmarks, BVPO improves over the best baseline by up to 7.8 points on AlpacaEval 2 and 6.8 points on Arena-Hard. Remarkably, despite being trained only on general conversational data, BVPO also enhances reasoning performance, boosting the base model's average performance across six math reasoning benchmarks by up to 4.0 points. These results establish trace sampling variance as a critical alignment challenge and demonstrate that directly optimizing the bias-variance trade-off yields both more stable training and stronger overall performance.

## Critique

Of course. Here is a commentary on the strengths and weaknesses of the paper "From Noisy Traces to Stable Gradients: Bias–Variance Optimized Preference Optimization for Aligning Large Reasoning Models".

### Strengths

1.  **High Novelty and Well-Defined Problem:** The paper tackles a very timely and under-explored problem: the alignment of Large Reasoning Models (LRMs). It clearly identifies a specific and significant bottleneck—the high gradient variance introduced by stochastic trace sampling—that is not present in standard LLM alignment. The core idea of mixing a high-variance trace-based gradient with a low-variance empty-trace gradient is elegant, intuitive, and novel in this context.

2.  **Strong Theoretical Foundation:** This is a major strength of the paper. The authors don't just propose a heuristic; they provide a rigorous theoretical analysis.
    *   **Theorem 1** formally proves the variance reduction property of their combined estimator.
    *   **Theorem 2** provides a closed-form, principled method for choosing the optimal mixing coefficient `α` by minimizing the Mean Squared Error (MSE), with a guarantee that their estimator is never worse than the best individual component.
    *   **Theorems 3 & 4** connect this statistical improvement directly to tighter convergence bounds for Stochastic Gradient Descent (SGD), creating a clear link from the method's design to its expected algorithmic performance.

3.  **Compelling and Comprehensive Empirical Results:** The experimental evaluation is thorough and convincing.
    *   **Alignment Performance:** The improvements on established benchmarks like AlpacaEval 2 and Arena-Hard are substantial (up to +7.8 points), demonstrating the practical effectiveness of BVPO.
    *   **Reasoning Preservation/Improvement:** A critical and impressive result is that BVPO not only preserves but *improves* reasoning performance on six math benchmarks, even though it was trained only on general conversational data. This directly addresses a key concern when aligning specialized models and significantly strengthens the paper's claim.
    *   **Model Scale:** Testing on three different model sizes (1.5B, 7B, 8B) shows the method's robustness and scalability.

4.  **Clarity of Presentation:** The paper is generally well-written. The structure is logical, moving from problem formulation to method description, theoretical analysis, and finally experiments. The use of clear notation (e.g., `g_t`, `g_e`, `g_c`, `L_m`, `L_t`) helps in following the technical arguments.

### Weaknesses

1.  **Limited Discussion of Practical `α` Selection:** While Theorem 2 provides a closed-form solution for the optimal `α`, it relies on population-level statistics (biases, covariances) that are unknown in practice. The paper does not explicitly detail how `α` was set in the experiments. Was it treated as a hyperparameter? Was the theoretical form approximated empirically? A brief discussion on the practical implementation of choosing `α` would strengthen the methodological clarity.

2.  **Ablation Studies and Sensitivity Analysis:** The paper would be even stronger with a more detailed ablation study.
    *   **`α` Sensitivity:** How sensitive are the results to the choice of `α`? A plot showing performance vs. `α` would be very informative.
    *   **Component Contribution:** An ablation directly comparing `L_t` (trace-only), `L_e` (empty-trace-only), and `L_c` (combined) on a learning curve would visually demonstrate the improved stability and final performance of BVPO.
    *   **Method Agnosticism:** The authors state BVPO is agnostic to the preference optimization algorithm, but they only instantiate it with DPO. Showing its application with another method like SimPO would solidify this claim.

3.  **Comparison to a Simple Baseline:** A natural baseline to include would be a version of DPO that uses multiple trace samples to reduce variance (e.g., using a multi-sample Monte Carlo estimator for the marginal loss). Comparing against such a baseline would help quantify the data efficiency and computational trade-off of BVPO versus simply "throwing more samples" at the problem.

4.  **Clarity on "Empty Trace" Implementation:** The description of how the empty-trace mode is implemented ("disabling reasoning trace generation by appending “<think></think>”") is clear from a systems perspective but slightly glosses over the modeling assumption. It assumes that `π_θ(r=∅, y|x)` is a meaningful and well-defined distribution that the model can learn, which is a valid approach but could be briefly discussed or justified.

### Overall Significance

This is a high-quality paper that makes a significant contribution. It identifies a clear and important problem in the emerging field of LRM alignment, proposes a simple yet theoretically grounded solution, and backs it up with extensive empirical evidence. The fact that the method improves both alignment *and* core reasoning capabilities is a particularly notable result. The work is likely to influence future research and practical efforts in aligning large-scale reasoning agents. The weaknesses are relatively minor and mostly concern additional analyses that could further bolster the already strong claims.

---

# LLM-Hanabi: Evaluating Multi-Agent Gameplays with Theory-of-Mind and Rationale Inference in Imperfect Information Collaboration Game

Authors: Fangzhou Liang, Tianshi Zheng, Chunkit Chan, Yauwai Yim, Yangqiu Song

Keywords: Mixture-of-Experts, Multilingual Routing, Cross-lingual Transfer, Sparse Activation, Language Specialization, Router Intervention, Multilingual Generalization

Comments: EMNLP 2025 Wordplay

Paper link: [http://arxiv.org/abs/2510.04980v1](http://arxiv.org/abs/2510.04980v1)

## Abstract

Effective multi-agent collaboration requires agents to infer the rationale behind others' actions, a capability rooted in Theory-of-Mind (ToM). While recent Large Language Models (LLMs) excel at logical inference, their ability to infer rationale in dynamic, collaborative settings remains under-explored. This study introduces LLM-Hanabi, a novel benchmark that uses the cooperative game Hanabi to evaluate the rationale inference and ToM of LLMs. Our framework features an automated evaluation system that measures both game performance and ToM proficiency. Across a range of models, we find a significant positive correlation between ToM and in-game success. Notably, first-order ToM (interpreting others' intent) correlates more strongly with performance than second-order ToM (predicting others' interpretations). These findings highlight that for effective AI collaboration, the ability to accurately interpret a partner's rationale is more critical than higher-order reasoning. We conclude that prioritizing first-order ToM is a promising direction for enhancing the collaborative capabilities of future models.

## Summary

This paper investigates multilingual routing behavior in Mixture-of-Experts (MoE) LLMs, revealing how sparse activation patterns affect cross-lingual generalization. The key contribution is demonstrating that MoE models exhibit language-specific routing in early and late layers but share language-universal experts in middle layers, mirroring patterns observed in dense LLMs. The authors establish a strong correlation between a language's performance and its routing alignment with English in these middle layers, showing that better-performing languages route more similarly to English.

The methodology involves two main components: interpretability analysis using parallel multilingual datasets to measure routing divergence, and intervention experiments that manipulate router logits during inference. For the analysis, the authors develop metrics to quantify cross-lingual routing divergence and entropy, revealing U-shaped patterns where divergence is lowest in middle layers. For interventions, they identify task-specific experts (using English domain data) and apply soft or hard interventions to steer non-English tokens toward these experts during evaluation.

The results show that activating English task experts in middle layers consistently improves multilingual performance across three state-of-the-art MoE models (Qwen3, Phi-3.5-MoE, GPT-OSS) and two evaluation tasks (MGSM and Global-MMLU medicine). Gains of 1-2% are statistically significant and particularly pronounced for lower-resource languages. Importantly, interventions are only effective when targeting middle layers and task experts—activating multilingual-specialized experts or intervening in other layers degrades performance.

This work provides the first comprehensive analysis of multilingual routing in MoE LLMs and demonstrates that cross-lingual expert sharing is a key driver of multilingual generalization, suggesting opportunities for improved training methods that enhance this alignment.

## Critique

Of course. Here is a commentary on the strengths and weaknesses of the paper "Multilingual Routing in Mixture-of-Experts".

### Overall Assessment

This is a strong, well-executed paper that makes a significant contribution to the interpretability of Mixture-of-Experts (MoE) models, specifically in the context of multilingualism. It successfully bridges findings from dense model interpretability with the unique architecture of MoEs, providing both novel insights and a practical intervention method.

---

### Strengths

1.  **High Novelty and Timeliness:** The core research question—*how do MoEs handle multilinguality internally?*—is highly novel. While the "U-shape" of language-specific vs. language-universal representations has been observed in dense models, demonstrating and quantifying this phenomenon through the lens of discrete expert routing in MoEs is a fresh and important contribution. The paper is timely given the industry's rapid adoption of MoE architectures.

2.  **Compelling and Multi-faceted Analysis:** The paper doesn't stop at a single observation. It builds a comprehensive case through a series of interconnected findings:
    *   **Finding 1 (U-shape):** Establishes the core phenomenon.
    *   **Finding 2 (Correlation with Performance):** Links the internal routing mechanics directly to a key external metric (task accuracy), making the analysis much more impactful.
    *   **Findings 3 & 4 (Entropy & Consistency):** Provide a deeper, more nuanced understanding of *how* routing differs for English vs. other languages, revealing that non-English tokens rely on fewer, more consistently activated experts.
    *   **Finding 5 (Language-Task Modularity):** This is a critical discovery that shows multilingual and task-specific experts are distinct sets, which directly enables the intervention strategy.

3.  **Significant and Actionable Results:** The intervention methodology is the paper's crowning achievement. Moving from correlation to causation, the authors demonstrate that simply steering the router to activate English-task experts during inference for non-English queries leads to consistent, statistically significant performance gains across multiple models and tasks. The fact that this works on fully trained, state-of-the-art models like Qwen and GPT-OSS is remarkable and suggests a clear pathway for future optimization.

4.  **Excellent Clarity and Presentation:**
    *   The writing is clear, logical, and easy to follow.
    *   The figures are well-designed and effectively illustrate the key concepts (e.g., the U-shape in Figure 2, the expert counts in Figure 5).
    *   The methodology for calculating routing divergence and identifying experts is described with precise mathematical formalism.
    *   The results are presented comprehensively in tables, showing both aggregated and per-language performance.

---

### Weaknesses

1.  **Limited Exploration of Causality in Analysis:** While the intervention proves a causal link between expert sharing and performance, some of the initial correlational findings could be probed deeper. For instance, the strong correlation between routing alignment and performance (Finding 2) might be partially confounded by the amount of pre-training data for each language. The authors mention language families as a confounder but a more rigorous ablation (e.g., controlling for data size) could strengthen this claim.

2.  **Scope and Generalizability of Interventions:**
    *   The improvements, while consistent, are modest (~1-2%). The paper rightly frames this as significant given the simplicity of the intervention, but it leaves open the question of how much room for improvement truly exists.
    *   The intervention is highly tailored, requiring model-specific tuning of layers, thresholds (`τ`), and intervention strength (`λ`). This limits its out-of-the-box applicability and suggests the underlying mechanism might be complex and brittle.
    *   The experiments are focused on knowledge-intensive tasks (math, medicine). It's unclear if the same intervention strategy would benefit more creative or open-ended generation tasks.

3.  **Unexplained Phenomena:** The paper notes but does not fully explain some curious observations, such as Phi-3.5-MoE's first layer activating the same few experts for all languages, which they attribute to "very poor load-balancing." A more in-depth hypothesis or investigation into why this occurs would have been interesting.

4.  **Potential Over-reliance on English as a Pivot:** The entire methodology is predicated on using English as the source of "good" expert activation patterns. This inherently reinforces English-centricity. While pragmatic, it would be valuable for future work to explore if experts from other high-resource languages could serve as effective pivots for linguistically related low-resource languages.

### Conclusion

This is an excellent paper that makes a substantial contribution to the field. Its primary strength lies in its successful translation of dense model interpretability concepts to the MoE setting, backed by a rigorous analysis and a compelling causal intervention. The weaknesses are minor and primarily point to exciting directions for future research rather than flaws in the current work. The findings have significant implications for both understanding and improving multilingual capabilities in the next generation of large language models.

---

# Multilingual Routing in Mixture-of-Experts

Authors: Lucas Bandarkar, Chenyuan Yang, Mohsen Fayyaz, Junlin Hu, Nanyun Peng

Keywords: Multi-Agent Reinforcement Learning, Tool-Integrated Planning, Policy Optimization, Large Language Models, Multi-Agent Systems

Comments: None

Paper link: [http://arxiv.org/abs/2510.04694v1](http://arxiv.org/abs/2510.04694v1)

## Abstract

Mixture-of-Experts (MoE) architectures have become the key to scaling modern LLMs, yet little is understood about how their sparse routing dynamics respond to multilingual data. In this work, we analyze expert routing patterns using parallel multilingual datasets and present highly interpretable layer-wise phenomena. We find that MoE models route tokens in language-specific ways in the early and late decoder layers but exhibit significant cross-lingual routing alignment in middle layers, mirroring parameter-sharing trends observed in dense LLMs. In particular, we reveal a clear, strong correlation between a model's performance in a given language and how similarly its tokens are routed to English in these layers. Extending beyond correlation, we explore inference-time interventions that induce higher cross-lingual routing alignment. We introduce a method that steers the router by promoting middle-layer task experts frequently activated in English, and it successfully increases multilingual performance. These 1-2% gains are remarkably consistent across two evaluation tasks, three models, and 15+ languages, especially given that these simple interventions override routers of extensively trained, state-of-the-art LLMs. In comparison, interventions outside of the middle layers or targeting multilingual-specialized experts only yield performance degradation. Altogether, we present numerous findings that explain how MoEs process non-English text and demonstrate that generalization is limited by the model's ability to leverage language-universal experts in all languages.

## Summary

Here is a summary of the paper "Multi-Agent Tool-Integrated Policy Optimization (MATPO)":

**Key Contributions:** This paper introduces MATPO, a novel reinforcement learning framework that enables training multiple agent roles (planner and worker agents) within a single large language model (LLM) instance. The key innovation is addressing the context length limitations and noisy tool responses that plague single-agent tool-integrated planning systems by decomposing tasks into planner-worker interactions while maintaining infrastructure efficiency.

**Methods:** MATPO extends Group Relative Policy Optimization (GRPO) to the multi-agent setting with a principled credit assignment mechanism across planner and worker rollouts. The framework uses role-specific system prompts to activate different agent behaviors within the same LLM, where the planner agent orchestrates high-level task decomposition and the worker agents handle specific subtasks (e.g., web searching). The method includes practical implementations like final summary mechanisms and user query recapping to improve inter-agent communication.

**Results:** Experiments on GAIA-text, WebWalkerQA, and FRAMES benchmarks demonstrate that MATPO consistently outperforms single-agent GRPO baselines by an average of 18.38% relative improvement. The multi-agent approach shows greater robustness to noisy tool outputs and more stable training progress. Ablation studies validate the importance of key components like final summaries and user query recapping, while also revealing practical insights about search result blocking and response formatting.

The work represents a significant step toward efficient multi-agent RL training that preserves the benefits of specialization while avoiding the infrastructure overhead of deploying multiple separate models.

## Critique

Of course. Here is a critique of the paper "Multi-Agent Tool-Integrated Policy Optimization (MATPO)" based on its strengths and weaknesses.

### Strengths

1.  **High Novelty and Clear Problem Formulation:** The paper tackles a genuinely novel and important problem: how to perform Reinforcement Learning (RL) on a multi-agent system where all agents are instantiated from a *single* Large Language Model (LLM). This "multi-agent-in-one-model" approach is a clever solution to the significant infrastructure and memory costs of deploying multiple separate models. The problem is well-motivated by the limitations of single-agent systems (context length, noise from tool outputs).

2.  **Principled and Well-Derived Methodology:** A key strength is the rigorous derivation of the MATPO algorithm from first principles. The paper clearly extends the single-agent Group Relative Policy Optimization (GRPO) objective to the multi-agent case, providing a principled credit assignment mechanism that shares the final task reward across both the planner and worker agent rollouts. This theoretical grounding is a significant contribution.

3.  **Significant and Robust Results:** The experimental results are compelling. An average relative improvement of **18.38%** over a strong single-agent GRPO baseline across three diverse benchmarks (GAIA-text, WebWalkerQA, FRAMES) is a substantial gain. The observation that MATPO is more stable and continues to improve where the single-agent baseline sometimes collapses is a powerful argument for its robustness.

4.  **Excellent Practical Insights:** The "Ablation Studies and Practical Take-Aways" section is exceptionally valuable. It moves beyond just reporting scores to provide actionable engineering insights (e.g., the necessity of final summaries, the mild impact of blocking certain URLs, the significant boost from "user query recapping"). This section greatly enhances the paper's practical utility for other researchers and engineers.

5.  **Clear Presentation and Implementation Details:** The paper is well-structured, with clear visualizations (Figures 1-4) that effectively illustrate the single-agent, multi-agent, and MATPO rollout processes. The inclusion of implementation details and the release of code promote reproducibility.

### Weaknesses

1.  **Limited Comparison to Multi-Agent-Multi-Model Baselines:** The most notable weakness is the lack of an empirical comparison to a "multi-agent-multi-model" baseline. While the paper argues convincingly for the infrastructure efficiency of the single-model approach, it does not demonstrate that MATPO's performance is comparable or superior to a system where the planner and worker are separate, fine-tuned models. This leaves an open question about the absolute performance ceiling.

2.  **Ablations Could Be More Quantitative:** While the ablation study is a strength, the results in Figure 6 are presented as trend lines without final numerical scores or statistical significance tests. A table summarizing the final performance of each ablated variant would make the conclusions more concrete and easier to digest.

3.  **Narrow Focus on Search/Web Tasks:** The experiments are confined to a specific class of tasks involving web search and scraping. While these are important, the paper's claims are broader. Demonstrating MATPO's effectiveness on tasks involving other tools (e.g., code execution, database querying, file processing) would significantly strengthen the generalizability of the approach. The future work section mentions this, but initial results would have been more impactful.

4.  **Potential Overhead and Complexity:** The method inherently introduces complexity. Each planner step triggers a full, multi-turn worker rollout, which can be computationally expensive and slow during both training and inference. The paper does not provide a analysis of the computational or latency overhead compared to the single-agent baseline.

### Summary

**Novelty:** High. The "multi-agent-in-one-model" RL formulation is a novel and timely contribution to the field of AI agents.

**Significance of Results:** Very High. The demonstrated performance gains and improved training stability are significant and directly address key pain points in tool-integrated LLM systems.

**Clarity of Presentation:** High. The paper is well-written, logically structured, and supported by clear diagrams and a valuable practical insights section.

In conclusion, this is a strong paper that introduces a novel, principled, and effective method for training multi-agent systems within a single LLM. Its main weaknesses lie in the scope of its empirical validation, but its core contributions are substantial and likely to influence future work in agent-based RL.

---

# Multi-Agent Tool-Integrated Policy Optimization

Authors: Zhanfeng Mo, Xingxuan Li, Yuntao Chen, Lidong Bing

Keywords: Partial Information Decomposition, Normalizing Flows, Gaussian Distributions, Multimodal Learning, Information Theory, Mutual Information

Comments: Work in progress

Paper link: [http://arxiv.org/abs/2510.04678v1](http://arxiv.org/abs/2510.04678v1)

## Abstract

Large language models (LLMs) increasingly rely on multi-turn tool-integrated planning for knowledge-intensive and complex reasoning tasks. Existing implementations typically rely on a single agent, but they suffer from limited context length and noisy tool responses. A natural solution is to adopt a multi-agent framework with planner- and worker-agents to manage context. However, no existing methods support effective reinforcement learning post-training of tool-integrated multi-agent frameworks. To address this gap, we propose Multi-Agent Tool-Integrated Policy Optimization (MATPO), which enables distinct roles (planner and worker) to be trained within a single LLM instance using role-specific prompts via reinforcement learning. MATPO is derived from a principled credit assignment mechanism across planner and worker rollouts. This design eliminates the need to deploy multiple LLMs, which would be memory-intensive, while preserving the benefits of specialization. Experiments on GAIA-text, WebWalkerQA, and FRAMES show that MATPO consistently outperforms single-agent baselines by an average of 18.38% relative improvement in performance and exhibits greater robustness to noisy tool outputs. Our findings highlight the effectiveness of unifying multiple agent roles within a single LLM and provide practical insights for stable and efficient multi-agent RL training.

## Summary

This paper introduces a new framework for Partial Information Decomposition (PID) that enables efficient and accurate estimation of information interactions in continuous, high-dimensional multimodal data. The key challenge addressed is that existing PID methods struggle with computational complexity and accuracy when dealing with continuous modalities, as they depend on optimizing over joint distributions constrained by estimated pairwise probabilities.

The paper makes three main contributions. First, it establishes that PID for Gaussian distributions (GPID) can be solved efficiently without loss of optimality when pairwise marginals are Gaussian, resolving an open question about joint Gaussian solutions. Second, it proposes Thin-PID, a gradient-based algorithm that significantly improves computational efficiency over existing methods by optimizing only the off-diagonal block of the noise covariance matrix, reducing complexity from O((d_X1+d_X2)³) to O(min(d_X1,d_X2)³). Third, the authors develop Flow-PID, which uses normalizing flows to transform arbitrary input distributions into latent Gaussian spaces while preserving mutual information, thus extending GPID's applicability to non-Gaussian data.

Experimental results demonstrate that Thin-PID achieves exact recovery of ground truth in Gaussian cases with errors <10⁻¹² and runs 10× faster than previous methods. Flow-PID shows superior performance on non-Gaussian synthetic examples and real-world multimodal benchmarks, providing more accurate quantification of redundant, unique, and synergistic information. The framework also proves effective for practical applications including dataset quantification and model selection, achieving over 96% of optimal performance when recommending models based on PID similarity.

The work addresses critical limitations in multimodal information analysis and provides both theoretical foundations and practical tools for understanding complex information interactions in high-dimensional data, with potential applications in guiding multimodal model development and data collection strategies.

## Critique

Of course. Here is a critique of the paper "Partial Information Decomposition via Normalizing Flows in Latent Gaussian Distributions," focusing on its strengths and weaknesses.

### Overall Summary

This paper presents a novel and computationally efficient framework, "Flow-PID," for estimating Partial Information Decomposition (PID) in high-dimensional, continuous data. It makes two key contributions: a theoretically-grounded, efficient algorithm for the Gaussian PID case ("Thin-PID") and a method to extend this to non-Gaussian data using normalizing flows. The paper is well-structured, the methodology is sound, and the results are significant, demonstrating clear improvements in speed and accuracy over existing baselines.

---

### Strengths

1.  **High Novelty and Strong Theoretical Foundation:**
    *   **Resolving an Open Problem:** The paper's first major contribution is proving that the optimal solution for Gaussian PID (GPID) is indeed a joint Gaussian distribution. This resolves the theoretical uncertainty in prior work (e.g., ~G-PID) and solidifies the foundation for all subsequent GPID algorithms.
    *   **Thin-PID Algorithm:** The derivation of Thin-PID is elegant. By reformulating the PID optimization within a Gaussian broadcast channel framework and optimizing only the off-diagonal noise covariance block, the authors achieve a significant reduction in computational complexity compared to methods requiring full eigenvalue decompositions.
    *   **Flow-PID Framework:** The idea of using a Cartesian product of normalizing flows to transform arbitrary data into a latent space with Gaussian marginals is innovative. The theoretical justification—that mutual information is preserved under these bijective mappings—is correct and powerful, allowing the efficient Thin-PID to be applied broadly.

2.  **Significant and Comprehensive Empirical Results:**
    *   **Validation on Synthetic Data:** The paper thoroughly validates its methods on synthetic data with known ground truth. The experiments show that Thin-PID is exact for Gaussian data and achieves superior accuracy and a **10x speedup** over the next-best method (Tilde-PID), especially as dimensions grow.
    *   **Generalization to Non-Gaussian Data:** The experiments on non-Gaussian synthetic data demonstrate that Flow-PID successfully recovers the true PID structure where Tilde-PID fails, validating the core premise of the framework.
    *   **Relevance to Real-World Problems:** The application to large-scale multimodal benchmarks (MultiBench) and specialized datasets (TCGA, VQA) is compelling. It moves beyond a theoretical exercise and shows the method's utility for quantifying modality interactions in practice and for model selection.

3.  **Clarity of Presentation:**
    *   The paper is generally well-written and logically organized. The progression from the Gaussian case (Thin-PID) to the general case (Flow-PID) is easy to follow.
    *   The use of theorems, lemmas, and propositions helps to clearly delineate the theoretical contributions.
    *   Figures and tables are effective in summarizing complex results, such as the complexity analysis and the comparison of PID values across different datasets and methods.

---

### Weaknesses

1.  **Technical Complexity and Accessibility:**
    *   The paper is highly technical, blending information theory, optimization, and deep learning. While the high-level idea is accessible, the full details of the Thin-PID derivation (e.g., the gradient in Proposition 3.4) and the Flow-PID objective will be challenging for readers without a strong background in all these areas. A higher-level intuition for the Thin-PID update steps could improve accessibility.

2.  **Limitations of the Flow-PID Approximation:**
    *   The authors correctly identify the main limitation: the accuracy of Flow-PID hinges on how well the normalizing flows can transform the true data marginals into Gaussians. The `Gaussian Marginal Loss` is a practical solution, but it is an approximation. The performance will degrade if the underlying data distribution is too complex or if the flow model is insufficiently powerful or trained with insufficient data. The paper could benefit from an ablation study on how flow architecture and dataset size affect the final PID estimate accuracy.

3.  **Validation on Real-World Data:**
    *   While the results on real-world data are interesting, the ultimate "ground truth" for PID in these datasets is unknown. The paper argues for plausibility (e.g., high synergy in VQA matches intuition), but this is inherently qualitative. A more rigorous, quantitative validation on a real-world(ish) task where the information structure is partially known would further strengthen the claims.

4.  **Computational Cost of Training Flows:**
    *   Although Thin-PID is fast, the overall Flow-PID pipeline requires training three separate normalizing flows. The paper does not discuss the computational cost of this training phase. For very large-scale applications, this upfront cost could be non-trivial and should be considered a part of the method's total computational footprint.

### Conclusion

This is a strong paper that makes substantial theoretical and practical contributions to the field of information-theoretic analysis in machine learning. By resolving a key theoretical question and introducing a highly efficient algorithm for the Gaussian case, it advances the state-of-the-art. The Flow-PID framework then creatively leverages this advancement to tackle the much more general and challenging problem of PID for arbitrary continuous data. The empirical validation is comprehensive and convincing. The main weaknesses relate to the inherent complexity of the method and the approximations introduced by the flow-based encoding, which the authors appropriately acknowledge. This work is likely to become a important reference for future research in multimodal learning and interpretability.

---

# Partial Information Decomposition via Normalizing Flows in Latent Gaussian Distributions

Authors: Wenyuan Zhao, Adithya Balachandran, Chao Tian, Paul Pu Liang

Keywords: Small Language Models, Model Orchestration, Multi-Agent Systems, Reasoning, Model Selection, Test-time Scaling, Confidence Estimation, Ensemble Methods

Comments: NeurIPS 2025

Paper link: [http://arxiv.org/abs/2510.04417v1](http://arxiv.org/abs/2510.04417v1)

## Abstract

The study of multimodality has garnered significant interest in fields where the analysis of interactions among multiple information sources can enhance predictive modeling, data fusion, and interpretability. Partial information decomposition (PID) has emerged as a useful information-theoretic framework to quantify the degree to which individual modalities independently, redundantly, or synergistically convey information about a target variable. However, existing PID methods depend on optimizing over a joint distribution constrained by estimated pairwise probability distributions, which are costly and inaccurate for continuous and high-dimensional modalities. Our first key insight is that the problem can be solved efficiently when the pairwise distributions are multivariate Gaussians, and we refer to this problem as Gaussian PID (GPID). We propose a new gradient-based algorithm that substantially improves the computational efficiency of GPID based on an alternative formulation of the underlying optimization problem. To generalize the applicability to non-Gaussian data, we learn information-preserving encoders to transform random variables of arbitrary input distributions into pairwise Gaussian random variables. Along the way, we resolved an open problem regarding the optimality of joint Gaussian solutions for GPID. Empirical validation in diverse synthetic examples demonstrates that our proposed method provides more accurate and efficient PID estimates than existing baselines. We further evaluate a series of large-scale multimodal benchmarks to show its utility in real-world applications of quantifying PID in multimodal datasets and selecting high-performing models.

## Summary

Here is a summary of the paper "SLM-MUX: Orchestrating Small Language Models for Reasoning":

**Key Problem & Contribution:** This paper identifies a critical limitation in existing LLM orchestration methods (e.g., Mixture-of-Agents, LLM-Debate) when applied to Small Language Models (SLMs). While these discussion-based methods work well for frontier LLMs by enabling them to correct each other, the authors show they actually *harm* SLM performance, causing accuracy drops of up to 5.5% due to error amplification and "groupthink." To address this, the paper introduces SLM-MUX, a novel orchestration framework specifically designed for SLMs that avoids explicit inter-model communication.

**Method:** The SLM-MUX framework operates in two main phases:
1. **Independent Generation:** Multiple SLMs independently generate several responses to the same query.
2. **Confidence Estimation & Selection:** Instead of discussing, the system selects the final answer based on which model shows the highest self-consistency (i.e., produces the same answer most frequently across its samples). Ties are broken using the models' validation accuracy.

The authors also introduce a **model selection search** strategy that systematically identifies complementary SLMs from a pool by optimizing for "Union Accuracy" (the percentage of questions at least one model can solve) while penalizing "Contradiction" (cases where an overconfident wrong answer suppresses a correct one). Furthermore, they explore **compute scaling strategies** by varying the number of models and samples per model.

**Key Results:**
- SLM-MUX significantly outperforms existing orchestration methods, achieving improvements of up to **13.4% on MATH, 8.8% on GPQA, and 7.0% on GSM8K**.
- With just two optimally selected SLMs, SLM-MUX **outperforms the much larger Qwen 2.5 72B model** on GPQA and GSM8K, and matches its performance on MATH.
- The model selection search proves crucial, as simply adding more models can be counterproductive without considering complementarity and contradiction penalties.
- The work provides both empirical evidence and mathematical analysis to explain why their method succeeds where others fail, positioning multi-SLM orchestration as a viable "multi-core" alternative to scaling ever-larger monolithic models.

## Critique

Of course. Here is a critique of the paper "SLM-MUX: Orchestrating Small Language Models for Reasoning," focusing on its strengths and weaknesses.

### Strengths

1.  **Clear Problem Identification and Motivation:** The paper excels at identifying a specific, important, and under-explored problem: the failure of existing "discussion-based" LLM orchestration methods (like Mixture-of-Agents, LLM-Debate) when applied to Small Language Models (SLMs). The analogy to multi-core processors as an alternative to scaling monolithic models is compelling and effectively frames the research.

2.  **Novelty of the Approach:** The core idea of SLM-MUX is simple yet powerful and well-motivated. Recognizing that SLMs are poor at critiquing each other's reasoning, the authors pivot to a non-communicative, confidence-based selection method. This is a novel and timely contribution to the field of model ensembling and orchestration, specifically tailored for the burgeoning class of SLMs.

3.  **Comprehensive and Significant Results:** The empirical evaluation is thorough and convincing.
    *   **Headline Results:** The key claim—that an ensemble of just two carefully selected SLMs can match or even outperform a massive model like Qwen 2.5 72B on challenging benchmarks (GPQA, GSM8K, MATH)—is a significant result. It provides a strong, cost-effective alternative to using frontier models.
    *   **Ablation and Comparison:** The paper systematically demonstrates the failure of baseline methods on SLMs and then shows SLM-MUX's consistent superiority. The exploration of two compute-scaling dimensions (number of models vs. samples per model) adds practical depth.

4.  **Insightful Model Selection Strategy:** The proposed model selection search, which optimizes for "Union Accuracy" while penalizing "Contradiction Penalty," is a sophisticated and crucial component. It moves beyond the naive assumption that simply adding the highest-performing individual models is optimal, addressing the real-world challenge of model complementarity and overconfident errors.

### Weaknesses

1.  **Limited Analysis of Confidence Estimation:** The method's primary weakness lies in its reliance on self-consistency as a proxy for confidence. The paper acknowledges this in the limitations section, but the analysis could be deeper. For instance, how often does a model produce a confidently wrong (highly self-consistent but incorrect) answer that "wins" the selection? A quantitative breakdown of such failure cases would strengthen the critique of the method's core mechanism.

2.  **Scalability of Model Selection:** The exhaustive search strategy for model selection, while effective for the 5-model pool used, does not scale. The paper correctly identifies this as a limitation, but it remains a significant practical hurdle for anyone wanting to apply this method to a larger zoo of available SLMs. A discussion of, or experiment with, a more scalable heuristic (even a simple greedy search) would have been valuable.

3.  **Clarity of Mathematical Analysis:** The mathematical analysis in Section 5, while well-intentioned, is somewhat simplistic and feels disconnected from the actual method. It primarily analyzes self-consistency and a straw-man "Agent Forest" baseline. A more direct theoretical analysis of SLM-MUX's selection rule and how it compares to an optimal selector would be more impactful.

4.  **Presentation and Figure Clarity:**
    *   **Referenced Figures:** The text frequently references figures (e.g., "Figure 1," "Figure 4") that are not included in the provided markdown, which hinders the reader's ability to fully assess the claims.
    *   **Table 1 Ambiguity:** In Table 1, the "Single-Best-SC" (Self-Consistency) baseline is extremely strong, even outperforming SLM-MUX on GPQA. This is a critical point that deserves more discussion. Why does a simple self-consistency on the single best model sometimes rival the more complex multi-model orchestration? This suggests the gains from SLM-MUX are highly dependent on the specific benchmark and model pool.

### Overall Assessment

This is a strong paper that makes a valuable contribution. It identifies a genuine gap in the literature and proposes a simple, effective, and well-evaluated solution. The results are significant, demonstrating that intelligent orchestration of smaller models is a viable and powerful alternative to using monolithic, large models. While the method has limitations, particularly regarding its confidence estimation and selection scalability, the core insight and empirical evidence are compelling. The presentation is generally clear, though the reliance on missing figures and the need for a deeper discussion of certain baseline comparisons are minor drawbacks.

