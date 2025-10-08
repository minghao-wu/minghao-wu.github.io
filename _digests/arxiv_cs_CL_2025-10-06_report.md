---
title: "ArXiv Daily Digest on 2025-10-06"
collection: digests
permalink: /digests/arxiv_cs_CL_2025-10-06_report
date: 2025-10-06
---

Today's research landscape showcases exciting advancements in multi-agent systems and model optimization, with several papers exploring how Large Language Models (LLMs) can collaborate more effectively. The theme of multi-agent collaboration appears prominently across multiple studies, including frameworks like MARS (Multi-Agent System for Deep ReSearch) and MATPO (Multi-Agent Tool-Integrated Policy Optimization), which demonstrate how specialized agent roles can enhance complex reasoning tasks. Another significant trend involves improving training efficiency through innovative approaches to combining Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL), with methods like MIFO (Mitigating Forgetting Between Supervised and Reinforcement Learning) showing remarkable data efficiency gains. In the multilingual domain, analysis of Mixture-of-Experts (MoE) architectures reveals fascinating routing patterns, while new benchmarks like LLM-Hanabi provide sophisticated ways to evaluate Theory-of-Mind (ToM) capabilities in collaborative settings. These developments collectively point toward more efficient, collaborative, and capable AI systems that better mimic human reasoning processes.

## TL;DR

Here's a TL;DR summary of the key themes and insights from these papers:

**Main Themes:**
- **Multi-agent collaboration & orchestration** - Multiple papers explore how to effectively coordinate multiple language models or agent roles for improved reasoning and task performance
- **Training optimization** - Focus on efficient training methods combining supervised fine-tuning (SFT) and reinforcement learning (RL) while mitigating catastrophic forgetting
- **Reasoning model alignment** - New methods for aligning large reasoning models with human preferences while managing gradient variance

**Key Insights:**
- **SLM-MUX** shows that orchestrating small language models can match or exceed performance of much larger models through confidence-based selection
- **MIFO** demonstrates that mitigating forgetting between SFT and RL yields stronger reasoners with dramatically improved data efficiency (using only 1.5% SFT data)
- **BVPO** addresses gradient variance in reasoning model alignment by optimizing bias-variance trade-off, improving both alignment and reasoning performance
- **Routing analysis** reveals that MoE models process multilingual content with language-specific routing in early/late layers but cross-lingual alignment in middle layers
- **Theory-of-Mind** capability strongly correlates with collaborative performance in multi-agent settings, with first-order ToM being more critical than higher-order reasoning

**Trend:** Move toward more efficient, collaborative AI systems that combine multiple specialized components rather than scaling monolithic models.

---

# From Noisy Traces to Stable Gradients: Bias-Variance Optimized Preference Optimization for Aligning Large Reasoning Models

Authors: Mingkang Zhu, Xi Chen, Bei Yu, Hengshuang Zhao, Jiaya Jia

Keywords: Large Reasoning Models, Preference Optimization, Bias-Variance Trade-off, Gradient Variance, Direct Preference Optimization, Trace Sampling, Alignment, Mathematical Reasoning

Comments: None

Paper link: [http://arxiv.org/abs/2510.05095v1](http://arxiv.org/abs/2510.05095v1)

## Abstract

Large reasoning models (LRMs) generate intermediate reasoning traces before producing final answers, yielding strong gains on multi-step and mathematical tasks. Yet aligning LRMs with human preferences, a crucial prerequisite for model deployment, remains underexplored. The statistically correct objective for preference alignment requires marginalizing over reasoning traces, but this computation is intractable in practice. A common workaround optimizes a single sampled trajectory, which introduces substantial gradient variance from stochastic trace sampling. To address this challenge, we frame preference optimization for LRMs through the lens of the bias--variance trade-off and propose Bias--Variance Optimized Preference Optimization (BVPO), a simple, drop-in method that mixes two gradient estimators: a high-variance trace-based estimator and a low-variance empty-trace estimator obtained by disabling reasoning trace generation. Our theory shows that BVPO strictly reduces trace-induced variance for any nontrivial mixture, provides a closed-form choice of the mixing weight that minimizes mean-squared error relative to the true marginal gradient, and under standard smoothness and step-size conditions, tightens classical convergence bounds for stochastic gradient descent. Empirically, BVPO improves alignment over the best baseline by up to 7.8 points on AlpacaEval~2 and 6.8 points on Arena-Hard. Despite being trained only on general conversational data, BVPO also boosts reasoning performance for base models by up to 4.0 points on the average of six math reasoning benchmarks. These results identify variance from trace sampling as a key bottleneck and demonstrate that directly optimizing the bias--variance trade-off yields more stable training and stronger overall performance.

## Summary

This paper addresses the challenge of aligning Large Reasoning Models (LRMs) with human preferences, identifying high gradient variance from stochastic reasoning trace sampling as a key bottleneck. While standard preference optimization methods like DPO work well for conventional LLMs, they struggle with LRMs due to the computational intractability of marginalizing over all possible reasoning traces, forcing practical implementations to rely on single sampled traces that introduce substantial gradient noise.

The key contribution is Bias–Variance Optimized Preference Optimization (BVPO), a simple yet principled method that combines two gradient estimators: a high-variance trace-based estimator and a low-variance empty-trace estimator (obtained by disabling reasoning trace generation). BVPO forms a convex combination of these estimators, with the mixing weight optimized to minimize Mean Squared Error relative to the ideal marginal gradient. Theoretically, the authors prove that BVPO strictly reduces trace-induced variance, provides a closed-form optimal mixing coefficient, and tightens SGD convergence bounds.

Empirical results demonstrate BVPO's effectiveness across three LRMs. On alignment benchmarks (AlpacaEval 2 and Arena-Hard), BVPO improves over the best baselines by up to 7.8 and 6.8 points respectively. Notably, despite being trained only on general conversational data, BVPO also enhances reasoning performance, boosting the average performance on six math reasoning benchmarks by up to 4.0 points. These results establish trace sampling variance as a critical alignment challenge and show that explicit bias–variance optimization yields both training stability and performance improvements.

## Critique

Of course. Here is a commentary on the strengths and weaknesses of the paper "From Noisy Traces to Stable Gradients: Bias–Variance Optimized Preference Optimization for Aligning Large Reasoning Models."

### Strengths

1.  **High Novelty and Timeliness:** The paper tackles a highly relevant and underexplored problem: the alignment of Large Reasoning Models (LRMs). While preference optimization (e.g., DPO) is well-established for standard LLMs, its application to LRMs that generate long, stochastic reasoning traces is a nascent area. The core insight—that trace sampling introduces debilitating gradient variance—is both novel and significant.

2.  **Elegant and Principled Solution:** The proposed method, BVPO, is conceptually simple, elegant, and grounded in solid statistical theory. The idea of mixing a high-variance trace-based gradient with a low-variance, deterministic "empty-trace" gradient is a direct and intuitive application of the bias-variance trade-off. Its "drop-in" nature makes it easy to adopt.

3.  **Strong Theoretical Foundation:** The paper is not just an empirical demonstration; it provides a rigorous theoretical analysis. Theorems on conditional variance reduction, MSE-optimal combination, and the direct link to tighter SGD convergence bounds are major strengths. This theoretical grounding elevates the work from a mere engineering trick to a principled contribution.

4.  **Comprehensive and Compelling Empirical Evaluation:** The experiments are thorough. The authors:
    *   Test on multiple model sizes (1.5B, 7B, 8B), demonstrating scalability.
    *   Evaluate on standard alignment benchmarks (AlpacaEval 2, Arena-Hard) and show significant improvements (up to +7.8 points).
    *   Crucially, evaluate on **mathematical reasoning benchmarks** to show that alignment on conversational data does not harm (and can even improve) core reasoning capabilities. This addresses a critical concern for deploying aligned LRMs.
    *   Evaluate models in both "Thinking" and "NoThinking" modes, showing the method's robustness.

5.  **Clarity of Presentation:** The paper is generally well-written. The structure is logical, moving from problem formulation to solution, theory, and experiments. The use of clear notation (e.g., \(g_t\), \(g_e\), \(g_c\), \(\mathcal{L}_m\), \(\mathcal{L}_t\)) and the step-by-step explanation of the theoretical motivation make the complex concepts accessible.

### Weaknesses

1.  **Limited Exploration of the Optimal \(\alpha\):** While Theorem 2 provides a closed-form solution for the optimal mixing weight \(\alpha^*\), the paper does not detail how this is computed or approximated in practice. The experiments seem to use a fixed \(\alpha\) (the value is not specified in the main text, though it might be in the appendix). A discussion on the practical challenges of estimating the biases and covariances for \(\alpha^*\), or an ablation study on the sensitivity to the choice of \(\alpha\), would have strengthened the practical contribution.

2.  **Baseline Comparison Could Be Broader:** The empirical comparison is primarily against DPO and SimPO. It would be valuable to see a comparison against other strategies for reducing variance, such as using multiple trace samples (a multi-sample Monte Carlo estimator) to see if the performance gains are due specifically to the bias-variance optimization or simply from using more compute to reduce variance.

3.  **Ablation Studies:** The paper would benefit from more ablation studies. For instance:
    *   How much of the performance gain comes from the empty-trace loss \(\mathcal{L}_e\) alone?
    *   An analysis of how the variance of the gradients (or the loss) actually changes during training with BVPO versus baselines would provide direct empirical support for the core thesis.

4.  **Clarity on the "Empty Trace":** The implementation of the "empty trace" (appending `<think></think>`) is mentioned, but it would be helpful to explicitly state that this is a feature supported by the specific LRMs (DeepSeek-R1) used in the experiments. A reader might wonder if this is a general technique or model-specific.

### Overall Assessment

This is a high-quality paper that makes a significant contribution to the field of aligning large generative models. It identifies a clear and important problem, proposes a simple yet theoretically sound solution, and backs it up with thorough experiments and rigorous theory. The weakness regarding the practical determination of \(\alpha\) is minor compared to the overall strength and novelty of the work. The results are highly significant, showing that it is possible to achieve better alignment **and** better reasoning simultaneously by explicitly managing the bias-variance trade-off inherent in training LRMs. This work is likely to influence both future research and practical deployment of reasoning models.

---

# Slm-mux: Orchestrating small language models for reasoning

Authors: Chenyu Wang, Zishen Wan, Hao Kang, Emma Chen, Zhiqiang Xie, Tushar Krishna, Vijay Janapa Reddi, Yilun Du

Keywords: Small Language Models, Model Orchestration, Multi-Agent Systems, Reasoning, Confidence Estimation, Model Selection, Compute Scaling, Test-time Scaling

Comments: None

Paper link: [http://arxiv.org/abs/2510.05077v1](http://arxiv.org/abs/2510.05077v1)

## Abstract

With the rapid development of language models, the number of small language models (SLMs) has grown significantly. Although they do not achieve state-of-the-art accuracy, they are more efficient and often excel at specific tasks. This raises a natural question: can multiple SLMs be orchestrated into a system where each contributes effectively, achieving higher accuracy than any individual model? Existing orchestration methods have primarily targeted frontier models (e.g., GPT-4) and perform suboptimally when applied to SLMs. To address this gap, we propose a three-stage approach for orchestrating SLMs. First, we introduce SLM-MUX, a multi-model architecture that effectively coordinates multiple SLMs. Building on this, we develop two optimization strategies: (i) a model selection search that identifies the most complementary SLMs from a given pool, and (ii) test-time scaling tailored to SLM-MUX. Our approach delivers strong results: Compared to existing orchestration methods, our approach achieves up to 13.4% improvement on MATH, 8.8% on GPQA, and 7.0% on GSM8K. With just two SLMS, SLM-MUX outperforms Qwen 2.5 72B on GPQA and GSM8K, and matches its performance on MATH. We further provide theoretical analyses to substantiate the advantages of our method. In summary, we demonstrate that SLMs can be effectively orchestrated into more accurate and efficient systems through the proposed approach.

## Summary

Here is a summary of the paper "SLM-MUX: Orchestrating Small Language Models for Reasoning":

**Key Problem and Contribution:** This paper identifies a critical limitation in existing LLM orchestration methods—they are designed for and perform well with frontier models like GPT-4 but actually *harm* performance when applied to Small Language Models (SLMs). The authors find that discussion-based methods (e.g., Mixture-of-Agents, LLM-Debate) cause SLMs to fall into "groupthink," amplifying errors rather than correcting them. To address this, they propose SLM-MUX, a novel orchestration framework specifically designed for SLMs that avoids explicit model discussions and instead selects outputs based on confidence scores.

**Proposed Method (SLM-MUX):** The framework operates in two phases: 1) **Independent Generation**, where each SLM independently generates multiple responses to a query, and 2) **Confidence Estimation**, where the most frequent answer from each model is selected, with its frequency serving as a confidence score. The final output is chosen from the model with the highest self-consistency (confidence), using validation accuracy as a tie-breaker. The authors also introduce a **Model Selection Search** strategy that systematically identifies complementary model subsets by maximizing union accuracy while penalizing cases where overconfident wrong answers suppress correct ones. Additionally, they explore **Compute Scaling Strategies** along two dimensions: adding more model types and drawing more samples per model.

**Key Results:** Experiments on MATH, GPQA, and GSM8K benchmarks demonstrate that SLM-MUX significantly outperforms existing orchestration methods, achieving improvements of up to 13.4% on MATH, 8.8% on GPQA, and 7.0% on GSM8K. With just two optimally selected SLMs, SLM-MUX matches or even surpasses the performance of the much larger Qwen 2.5 72B model—outperforming it on GPQA and GSM8K and matching its performance on MATH. The model selection search and compute scaling strategies are shown to provide substantial additional performance gains, validating the importance of complementary model pairing and careful resource allocation.

In summary, this work establishes that a "multi-core" approach of intelligently orchestrating multiple efficient SLMs is a viable and promising alternative to scaling ever-larger monolithic models, opening a new direction for building capable and cost-effective AI systems.

## Critique

Of course. Here is a critique of the paper "SLM-MUX: Orchestrating Small Language Models for Reasoning," focusing on its strengths and weaknesses.

### Strengths

1.  **Clear Problem Identification and Motivation:** The paper excels at identifying a specific, important, and counter-intuitive problem: existing discussion-based LLM orchestration methods (like Mixture-of-Agents, LLM-Debate) fail when applied to Small Language Models (SLMs). The analogy to multi-core processors is compelling and effectively frames the research direction.

2.  **Novelty of the Approach:** The core idea of SLM-MUX is simple, elegant, and well-motivated. Instead of forcing SLMs to "discuss" (a process where they tend to amplify errors), it uses a confidence-based selection mechanism (self-consistency) to pick the best answer from a pool of independently generated ones. This is a pragmatic shift in perspective for the SLM domain. The accompanying **model selection search** is a valuable contribution that moves beyond naive model aggregation by explicitly optimizing for complementary strengths and penalizing overconfident contradictions.

3.  **Significant and Well-Evaluated Results:** The empirical results are a major strength. The paper demonstrates:
    *   **Clear Failure of Baselines:** It systematically shows that existing methods harm SLM performance.
    *   **Strong Performance of SLM-MUX:** It achieves substantial improvements (e.g., +13.4% on MATH) over both single models and other orchestration methods.
    *   **Competitiveness with Large Models:** The headline result—that a 2-SLM ensemble can outperform or match a 72B parameter model (Qwen 2.5 72B)—is significant and validates the "multi-core" thesis. The ablation studies on model selection and compute scaling provide a thorough understanding of the method's behavior.

4.  **Clarity of Presentation:** The paper is generally well-structured and easy to follow. The use of algorithms (Algorithm 1), figures (visualizing workflows, results, and trade-offs), and tables makes the methodology and findings accessible.

### Weaknesses

1.  **Limited Analysis of "Why" Discussion Fails:** While the paper identifies the "groupthink" problem, the analysis in the main text is somewhat superficial. The claim that SLMs lack the reasoning ability to correct each other is plausible but not deeply investigated. A more thorough qualitative analysis of failure cases in the discussion-based methods (e.g., how correct answers get overturned) would have strengthened this key argument. The relegation of this analysis to an appendix is a minor weakness.

2.  **Simplicity and Potential Limitations of the Core Method:** The strength of SLM-MUX is its simplicity, but this is also a potential weakness. The paper acknowledges two key limitations:
    *   **Static Framework:** The model selection is fixed per benchmark and does not adapt to the specific query. A per-instance routing mechanism could be a more powerful (though more complex) future direction.
    *   **Confidence Metric:** Relying solely on self-consistency as a confidence measure is a known weakness, as models can be consistently wrong. The "Contradiction Penalty" in the model search indirectly addresses this, but the core method itself remains vulnerable to this issue.

3.  **Theoretical Analysis Could Be Stronger:** The mathematical analysis in Section 5 is a good start but feels somewhat disconnected from the main method. It primarily explains why self-consistency works for a single model under certain conditions and why Agent Forest (which averages over models) is inferior to selecting the best model's output. A more formal analysis framing the problem as a selection rule among multiple, potentially correlated, agents would have been more impactful.

4.  **Scope of Evaluation:** The evaluation is strong but focused exclusively on reasoning-heavy, multiple-choice-style benchmarks (MATH, GPQA, GSM8K). It remains to be seen how well SLM-MUX generalizes to more open-ended tasks like creative writing, summarization, or complex instruction-following, where the concept of a "correct answer" is less well-defined and self-consistency may be a less reliable signal.

### Overall Assessment

This is a strong paper that makes a clear and valuable contribution. It identifies a genuine problem in the current research landscape and proposes a simple, effective, and well-evaluated solution. The significance of the results—showing that small, efficiently orchestrated models can compete with much larger monolithic models—is high and has practical implications for cost-effective AI deployment. While the analysis of the core problem and the theoretical underpinnings could be deepened, the empirical evidence is compelling and the core ideas are novel and impactful. The presentation is clear and effectively communicates the research.

---

# LLM-Hanabi: Evaluating Multi-Agent Gameplays with Theory-of-Mind and Rationale Inference in Imperfect Information Collaboration Game

Authors: Fangzhou Liang, Tianshi Zheng, Chunkit Chan, Yauwai Yim, Yangqiu Song

Keywords: Multi-Agent Collaboration, Theory-of-Mind, Rationale Inference, Imperfect Information Games, Large Language Model Evaluation, Hanabi Benchmark

Comments: EMNLP 2025 Wordplay

Paper link: [http://arxiv.org/abs/2510.04980v1](http://arxiv.org/abs/2510.04980v1)

## Abstract

Effective multi-agent collaboration requires agents to infer the rationale behind others' actions, a capability rooted in Theory-of-Mind (ToM). While recent Large Language Models (LLMs) excel at logical inference, their ability to infer rationale in dynamic, collaborative settings remains under-explored. This study introduces LLM-Hanabi, a novel benchmark that uses the cooperative game Hanabi to evaluate the rationale inference and ToM of LLMs. Our framework features an automated evaluation system that measures both game performance and ToM proficiency. Across a range of models, we find a significant positive correlation between ToM and in-game success. Notably, first-order ToM (interpreting others' intent) correlates more strongly with performance than second-order ToM (predicting others' interpretations). These findings highlight that for effective AI collaboration, the ability to accurately interpret a partner's rationale is more critical than higher-order reasoning. We conclude that prioritizing first-order ToM is a promising direction for enhancing the collaborative capabilities of future models.

## Summary

Of course. Here is a summary of the paper "LLM-Hanabi: Evaluating Multi-Agent Gameplays with Theory-of-Mind and Rationale Inference in Imperfect Information Collaboration Game."

### Summary

This paper introduces **LLM-Hanabi**, a novel benchmark designed to evaluate the Theory-of-Mind (ToM) and rationale inference capabilities of Large (Language) Models in a dynamic, collaborative setting. The key idea is to move beyond static, text-based ToM evaluations and assess how well AI agents can reason about each other's intentions and beliefs during an interactive task.

**Key Contributions & Methods:**
The authors use the cooperative card game **Hanabi** as their testbed, as its core mechanics—where players hold hidden cards and must communicate via sparse, linguistic hints—naturally require inferring a partner's rationale. The LLM-Hanabi framework features an automated evaluation system that operates in two phases:
1.  **During gameplay**, agents are prompted to generate structured reasoning statements: the *Rationale* behind a hint (ground truth), the recipient's *First-Order ToM* (interpreting the hint's intent), and the hinter's *Second-Order ToM* (predicting how the recipient will interpret it).
2.  **Post-game**, an LLM-as-a-judge scores the alignment between these statements to produce quantitative First-Order and Second-Order ToM scores.

**Key Results:**
The evaluation of a diverse set of LLMs and larger reasoning models (LRMs) yielded several important findings:
*   **LRMs generally outperformed standard LLMs** in both game performance and ToM capabilities.
*   There is a **strong positive correlation between an agent's ToM score and its in-game success**.
*   Most notably, **First-Order ToM (interpreting a partner's intent) was a significantly stronger predictor of game performance** than Second-Order ToM (predicting a partner's interpretation), with correlation coefficients of r=0.76 vs. r=0.58.

**Conclusion:**
The paper concludes that for developing effective collaborative AI, the primary focus should be on enhancing an agent's ability to accurately **interpret its partners' actions and rationale** (first-order ToM), rather than on more complex higher-order reasoning. The LLM-Hanabi benchmark provides a scalable tool for pursuing this direction.

## Critique

Of course. Here is a critique of the paper "LLM-Hanabi: Evaluating Multi-Agent Gameplays with Theory-of-Mind and Rationale Inference in Imperfect Information Collaboration Game".

### Strengths

1.  **Strong and Well-Motivated Benchmark:** The choice of Hanabi is excellent. It is a well-established game in AI research for studying cooperation under imperfect information. The paper correctly identifies the limitations of static, text-based ToM evaluations and effectively argues that Hanabi provides a dynamic, interactive, and scalable alternative. The focus on a purely cooperative game (as opposed to adversarial ones like Poker) cleanly isolates the challenge of collaborative reasoning.

2.  **Novel and Automated Evaluation Framework:** The core contribution—the automated ToM evaluation system—is novel and practical. By prompting agents to generate structured rationales and first/second-order ToM statements during gameplay and then using an LLM-as-a-judge to score them, the authors create a scalable method for quantifying a previously qualitative and hard-to-measure capability. This is a significant step beyond simply measuring win rates.

3.  **Significant and Actionable Findings:** The results are compelling and provide clear, actionable insights for the field. The strong positive correlation between ToM and game performance validates the benchmark's premise. The most impactful finding is that **first-order ToM is a stronger predictor of success than second-order ToM**. This challenges any assumption that more complex, recursive reasoning is always better and provides a clear direction for future model development: prioritize accurate intent interpretation.

4.  **Comprehensive Evaluation:** The paper benchmarks a wide range of models, including both standard LLMs and the newer "Large Reasoning Models" (LRMs), across multiple performance dimensions (game score, rounds, 1st/2nd-order ToM). This provides a rich, comparative landscape of the current state of the art.

### Weaknesses

1.  **Clarity on "LRM" and Reasoning Methods:** The distinction between "LLM" and "LRM" is not clearly defined in the main text. While it's implied that LRMs use "Long-CoT," the paper would benefit from a more explicit explanation of what constitutes an LRM, how their reasoning process differs architecturally or in terms of prompting, and why this leads to superior performance. The conflation of model type (LLM vs. LRM) with reasoning method (CoT vs. Long-CoT) in Table 1 is slightly confusing.

2.  **Potential Circularity in ToM Scoring:** A major methodological weakness is the reliance on the agent's own generated "Rationale" as the ground truth for scoring its partner's First-Order ToM. This creates a potential for circular reasoning: an agent could generate a poor or post-hoc rationale, which would then make it impossible for its partner to score highly on First-Order ToM, even if the partner made a correct and useful inference. The scoring system measures alignment with the stated rationale, not necessarily the correctness or optimality of the reasoning itself.

3.  **Limited Analysis of Failure Modes:** The results are presented at a high level (scores and correlations). The paper would be strengthened by a qualitative analysis of *why* models fail. For instance, what are common errors in Second-Order ToM? Are there specific game situations that consistently challenge all models? Providing examples of poor rationales or misinterpretations would make the findings more concrete and informative for developers.

4.  **Presentation and Proofreading:** The paper has several typographical and formatting issues (e.g., "LRM (Long-CoT)" in Table 1, broken figure references like `[1](https://arxiv.org/html/2510.04980v1#S1.F1)`, inconsistent bolding in the table caption). While minor, these errors detract from the professionalism and clarity of the presentation.

### Summary

This paper presents a strong, timely, and valuable contribution to the field of multi-agent AI and LLM evaluation. Its primary strength lies in the novel LLM-Hanabi benchmark and its automated ToM scoring system, which enables the discovery of the critical relationship between first-order reasoning and collaborative success. The main weaknesses are a lack of clarity around the "LRM" concept and a potentially problematic reliance on self-reported rationales as ground truth. Despite these issues, the core findings are significant and the benchmark itself is likely to be adopted and built upon by the research community.

---

# MARS: Optimizing Dual-System Deep Research via Multi-Agent Reinforcement Learning

Authors: Guoxin Chen, Zile Qiao, Wenqing Wang, Donglei Yu, Xuanzhong Chen, Hao Sun, Minpeng Liao, Kai Fan, Yong Jiang, Penguin Xie, Wayne Xin Zhao, Ruihua Song, Fei Huang

Keywords: Multi-Agent Reinforcement Learning, Dual-System Reasoning, Retrieval-Augmented Generation, Deep Research, Tool-Augmented Reasoning, Large Reasoning Models

Comments: Ongoing Work

Paper link: [http://arxiv.org/abs/2510.04935v1](http://arxiv.org/abs/2510.04935v1)

## Abstract

Large Reasoning Models (LRMs) often exhibit a tendency for overanalysis in simple tasks, where the models excessively utilize System 2-type, deliberate reasoning, leading to inefficient token generation. Furthermore, these models face challenges in adapting their reasoning capabilities to rapidly changing environments due to the static nature of their pretraining data. To address these issues, advancing Large Language Models (LLMs) for complex reasoning tasks requires innovative approaches that bridge intuitive and deliberate cognitive processes, akin to human cognition's dual-system dynamic. This paper introduces a Multi-Agent System for Deep ReSearch (MARS) enabling seamless integration of System 1's fast, intuitive thinking with System 2's deliberate reasoning within LLMs. MARS strategically integrates multiple external tools, such as Google Search, Google Scholar, and Python Interpreter, to access up-to-date information and execute complex computations, while creating a specialized division of labor where System 1 efficiently processes and summarizes high-volume external information, providing distilled insights that expand System 2's reasoning context without overwhelming its capacity. Furthermore, we propose a multi-agent reinforcement learning framework extending Group Relative Policy Optimization to simultaneously optimize both systems with multi-turn tool interactions, bin-packing optimization, and sample balancing strategies that enhance collaborative efficiency. Extensive experiments demonstrate MARS achieves substantial improvements of 3.86% on the challenging Humanity's Last Exam (HLE) benchmark and an average gain of 8.9% across 7 knowledge-intensive tasks, validating the effectiveness of our dual-system paradigm for complex reasoning in dynamic information environments.

## Summary

Of course. Here is a summary of the paper "MARS: Optimizing Dual-System Deep Research via Multi-Agent Reinforcement Learning."

**Key Contributions & Method:**
This paper introduces MARS, a novel multi-agent framework that enhances the reasoning capabilities of Large Language Models (LLMs) by integrating the dual-process theory of human cognition. The core innovation is the seamless collaboration between two systems within a single LLM: **System 2** handles deliberate, multi-step reasoning and strategically invokes external tools (Google Search, Google Scholar, Python Interpreter), while **System 1** acts as an efficient information processor, using fast, intuitive thinking to distill and summarize the often lengthy and noisy outputs from these tools. This division of labor prevents System 2 from being overwhelmed by large volumes of raw data, allowing it to focus on complex reasoning.

To train this system, the authors propose a multi-agent Reinforcement Learning (RL) framework that extends Group Relative Policy Optimization (GRPO) to optimize both systems simultaneously. Key technical strategies include:
1.  **Bin-Packing Optimization:** Efficiently organizes variable-length tool outputs (e.g., multiple web pages) into optimally-sized chunks for parallel processing by System 1.
2.  **Balanced Sampling Mechanism:** Pre-computes advantage signals and balances the number of training samples between the two systems to prevent one from dominating the learning process.

**Key Results:**
Extensive experiments demonstrate MARS's effectiveness on highly challenging benchmarks:
*   **Humanity’s Last Exam (HLE):** MARS achieves a substantial improvement of **3.86%** over its base model (Qwen2.5-7B), reaching **7.38%** accuracy. It outperforms all other open-source reasoning methods, including those based on much larger models (e.g., 32B and 72B parameters), and significantly narrows the performance gap with powerful proprietary systems.
*   **Knowledge-Intensive QA Tasks:** On a suite of 7 single-hop and multi-hop question-answering datasets, MARS achieves an average performance gain of **8.9%** over the previous state-of-the-art method, with particularly strong improvements on complex multi-hop reasoning tasks.

The paper provides comprehensive analyses, including an ablation study confirming the complementary value of different tools and an examination of the training dynamics, which show the model learning to use more tools and generate more sophisticated responses over time.

## Critique

Of course. Here is a critique of the paper "MARS: Optimizing Dual-System Deep Research via Multi-Agent Reinforcement Learning."

### Overall Summary

This paper presents a well-executed and compelling study on enhancing complex reasoning in LLMs. The core idea is both intuitive and powerful, and the empirical results are substantial, demonstrating clear improvements over strong baselines. The presentation is generally clear, though it has some minor organizational weaknesses.

---

### Strengths

1.  **High Novelty and Conceptual Elegance:**
    *   The central idea of formalizing a "dual-system" architecture within a single LLM, inspired by cognitive theory, is highly novel. The division of labor—System 2 for deliberate reasoning/planning and System 1 for efficient, high-volume information distillation—is an elegant solution to the problem of context overload in complex RAG tasks.
    *   The integration of this dual-system concept with a **multi-agent RL framework** is a significant contribution, moving beyond simple prompting or supervised fine-tuning to actively optimize the collaborative behavior between the two "agents."

2.  **Significant and Comprehensive Empirical Results:**
    *   The results are the paper's strongest asset. Achieving state-of-the-art performance on the extremely challenging **Humanity's Last Exam (HLE)** benchmark and a large average gain (+8.9%) across 7 other knowledge-intensive tasks is a major accomplishment.
    *   The fact that MARS, based on a 7B/8B parameter model, outperforms methods using models up to 72B parameters and even narrows the gap with some proprietary models is a powerful demonstration of its efficiency and effectiveness.
    *   The ablation study on tools is thorough and provides valuable insights into the contribution of each component (Python, Search, Scholar) across different domains.

3.  **Thoughtful Engineering and Methodological Rigor:**
    *   The proposed optimizations are practical and well-motivated:
        *   **Bin-Packing for System 1:** This is a clever engineering solution to a real-world problem (processing variable-length documents efficiently), directly contributing to the system's scalability.
        *   **Balanced Sampling Mechanism:** Addressing the sample imbalance between System 1 and System 2 during RL training is a crucial detail that likely prevents the training dynamics from being dominated by one system.
    *   The analysis section (Figure 3) is excellent, providing a transparent look into the training dynamics, including score progression, tool usage evolution, and response length changes.

---

### Weaknesses

1.  **Clarity and Presentation:**
    *   **Repetitive Introduction/Contributions:** The abstract and introduction are somewhat repetitive, with the contributions section largely rephrasing points already made. This could be condensed for better flow.
    *   **Methodology Section Complexity:** While the dual-system framework (Section 2.1) is well-explained, the jump to the RL optimization strategies (Section 2.2) is quite abrupt. A brief high-level overview of why RL is the chosen optimization paradigm before diving into the specifics would improve readability for a broader audience.
    *   **Figure Quality:** The figures in the paper (e.g., Figures 1 and 2) are described as "Refer to caption" in this text-only version, but in a final paper, their clarity is paramount. Figure 2, which outlines the complex RL process, must be exceptionally clear to be understood.

2.  **Technical and Experimental Limitations:**
    *   **Limited Comparison with Top Proprietary Systems:** While the results are impressive, the comparison with the strongest proprietary systems like "OpenAI Deep Research" is incomplete, as only the average score is reported. A full subject-by-subject comparison would provide a more precise benchmark.
    *   **Computational Cost Omission:** The paper does not discuss the significant computational and API costs associated with the training and inference process (e.g., the cost of multiple Google Search/Scholar API calls per question, and the RL training over thousands of steps). An analysis of cost-to-performance would be valuable for practical adoption.
    *   **Baseline Implementation Details:** It is not entirely clear if all baselines (like Search-R1, C-3PO) were re-implemented by the authors or taken from prior literature. If re-implemented, ensuring exact reproducibility is a challenge; if taken from other papers, the experimental conditions may not be perfectly aligned.

3.  **Conceptual Scope:**
    *   The "Dual-System" metaphor is powerful but could be explored further. The paper focuses on the division of labor (reasoning vs. summarization), but a deeper discussion on how this architecture specifically mitigates the stated problem of LRM "overanalysis" on simple tasks would strengthen the narrative.

---

### Conclusion

This is a strong paper with a highly novel core idea that is backed by significant empirical results. The proposed MARS framework represents a meaningful advance in building capable and efficient reasoning systems that can leverage large volumes of external information. Its main weaknesses lie in the presentation, which could be streamlined for greater clarity, and a more thorough discussion of computational costs and comparisons with the very best proprietary models. Overall, it is a solid contribution that likely represents a step forward in the field of agentic reasoning systems.

---

# Multilingual Routing in Mixture-of-Experts

Authors: Lucas Bandarkar, Chenyuan Yang, Mohsen Fayyaz, Junlin Hu, Nanyun Peng

Keywords: Multi-Agent Reinforcement Learning, Tool-Integrated Planning, Policy Optimization, Multi-Agent Systems, Language Model Training

Comments: None

Paper link: [http://arxiv.org/abs/2510.04694v1](http://arxiv.org/abs/2510.04694v1)

## Abstract

Mixture-of-Experts (MoE) architectures have become the key to scaling modern LLMs, yet little is understood about how their sparse routing dynamics respond to multilingual data. In this work, we analyze expert routing patterns using parallel multilingual datasets and present highly interpretable layer-wise phenomena. We find that MoE models route tokens in language-specific ways in the early and late decoder layers but exhibit significant cross-lingual routing alignment in middle layers, mirroring parameter-sharing trends observed in dense LLMs. In particular, we reveal a clear, strong correlation between a model's performance in a given language and how similarly its tokens are routed to English in these layers. Extending beyond correlation, we explore inference-time interventions that induce higher cross-lingual routing alignment. We introduce a method that steers the router by promoting middle-layer task experts frequently activated in English, and it successfully increases multilingual performance. These 1-2% gains are remarkably consistent across two evaluation tasks, three models, and 15+ languages, especially given that these simple interventions override routers of extensively trained, state-of-the-art LLMs. In comparison, interventions outside of the middle layers or targeting multilingual-specialized experts only yield performance degradation. Altogether, we present numerous findings that explain how MoEs process non-English text and demonstrate that generalization is limited by the model's ability to leverage language-universal experts in all languages.

## Summary

Here is a summary of the paper "Multi-Agent Tool-Integrated Policy Optimization (MATPO)":

**Key Contributions:** This paper introduces MATPO, a novel reinforcement learning framework that enables multiple agent roles (planner and worker) to be trained within a single LLM instance. The key innovation is addressing the credit assignment problem in multi-agent RL by developing a principled approach that allows both planner and worker agents to share responsibility for the final accuracy reward. This eliminates the need for deploying multiple LLMs while preserving the benefits of specialization.

**Methods:** MATPO extends single-agent Group Relative Policy Optimization (GRPO) to the multi-agent setting by modifying the policy gradient objective to account for both planner-agent rollouts and worker-agent rollouts. The framework uses a two-agent system where: 1) a planner-agent orchestrates high-level planning and delegates subtasks, and 2) worker-agents handle specific browsing tasks using search tools. Crucially, both roles are implemented using the same LLM with different system prompts. The implementation builds on existing single-agent RL frameworks and includes practical components like final-summary mechanisms and user query recapping to improve performance.

**Results:** Experiments on GAIA-text, WebWalkerQA, and FRAMES benchmarks show that MATPO consistently outperforms single-agent GRPO baselines, achieving average relative improvements of 18.38%. MATPO demonstrates greater robustness to noisy tool outputs and more stable training progress compared to single-agent approaches. Ablation studies reveal that key components like final summaries and user query recapping significantly contribute to performance gains, while practical insights highlight the importance of blocking sensitive URLs and improving response formats between agents.

The work demonstrates that unifying multiple agent roles within a single LLM can effectively address context length limitations and noise issues in tool-integrated planning while maintaining infrastructure efficiency.

## Critique

Of course. Here is a critique of the paper "Multi-Agent Tool-Integrated Policy Optimization" (MATPO).

### Overall Summary

This is a strong and timely paper that addresses a clear gap in the field of tool-using language agents. The core idea—training a single LLM to perform multiple agent roles (planner and worker) via reinforcement learning—is novel, practical, and well-executed. The paper presents a solid theoretical foundation, compelling empirical results, and valuable practical insights.

---

### Strengths

1.  **High Novelty and Clear Problem Formulation:** The paper identifies a significant limitation in single-agent tool-integrated planning (TIP): context window saturation and noise from tool outputs. The proposed solution—a multi-agent-in-one-model framework trained with RL—is a genuinely novel contribution. It elegantly addresses the infrastructure and parameter inefficiency of using separate models for each agent.

2.  **Principled Methodology:** The derivation of the MATPO objective from the multi-agent policy gradient is rigorous. The extension of the Group Relative Policy Optimization (GRPO) algorithm to handle credit assignment across a planner and multiple worker rollouts is a key technical contribution. The paper clearly explains how rewards are assigned and normalized across the different agent trajectories within a single model.

3.  **Significant and Robust Results:** The experimental results are convincing. An average relative improvement of **18.38%** over a strong single-agent GRPO baseline across three diverse benchmarks (GAIA-text, WebWalkerQA, FRAMES) is a substantial result. The finding that MATPO is more stable and avoids the performance collapse sometimes seen in single-agent training is a major point in its favor, highlighting a key advantage of the multi-agent decomposition.

4.  **Excellent Practical Insights:** The "Ablation Studies and Practical Take-Aways" section is a standout strength. It moves beyond simply reporting scores and provides actionable advice for practitioners, such as:
    *   The necessity of final summaries for worker agents.
    *   The mild impact of blocking certain URLs (HuggingFace).
    *   The crucial importance of "user query recapping" for the worker agent.
    *   The insightful observation about the "user message" format potentially biasing the planner, which is a great point for future work.

5.  **Clear Presentation and Implementation Details:** The paper is well-structured and the figures (1, 2, 3, 4) are effective in visualizing the different frameworks and the training process. The authors have provided code and detailed their implementation on top of an existing RL framework (veRL), which enhances reproducibility.

---

### Weaknesses and Potential Improvements

1.  **Limited Comparison to a "Multi-Agent-Multi-Model" Baseline:** While the focus on the "in-one-model" approach is justified for efficiency, the paper would be even stronger if it included a comparison to a "multi-agent-multi-model" baseline. This would help isolate the benefit of the *architectural decomposition* from the benefit of *training a single model on multiple roles*. Does the performance gain come from the multi-agent structure itself, or from the multi-role training?

2.  **Computational Cost Analysis:** The method is more parameter-efficient than deploying multiple models, but it is undoubtedly more computationally expensive during training and rollouts than single-agent GRPO due to the nested worker rollouts. A brief discussion or analysis of the increased computational/token cost would provide a more complete picture of the method's trade-offs.

3.  **Ablation on the Core Idea:** The most critical ablation would be to test a version of MATPO where the planner and worker are the same model but are **not jointly trained with RL** (i.e., using a base or SFT model for the worker). This would directly test the hypothesis that "RL training can benefit the model when it is exposed to experience from multiple agent roles."

4.  **Clarity on the "Single Model":** While the concept is clear, the paper could be more explicit upfront about the fact that the same model weights are used for both planner and worker, differentiated *only* by the system prompt. This is a subtle but crucial point that could be emphasized earlier.

---

### Conclusion

This paper makes a valuable contribution to the field of AI agents and reinforcement learning for LLMs. The proposed MATPO framework is novel, effective, and practically useful. Its strengths in problem formulation, methodological rigor, empirical results, and practical guidance far outweigh its minor weaknesses. The work successfully demonstrates that a single LLM can be trained to effectively play multiple collaborative roles, opening a promising direction for future research into more complex and capable agentic systems.

---

# Multi-Agent Tool-Integrated Policy Optimization

Authors: Zhanfeng Mo, Xingxuan Li, Yuntao Chen, Lidong Bing

Keywords: Multilingual Routing, Mixture-of-Experts (MoE), Cross-lingual Transfer, Expert Activation, Inference-time Intervention

Comments: Work in progress

Paper link: [http://arxiv.org/abs/2510.04678v1](http://arxiv.org/abs/2510.04678v1)

## Abstract

Large language models (LLMs) increasingly rely on multi-turn tool-integrated planning for knowledge-intensive and complex reasoning tasks. Existing implementations typically rely on a single agent, but they suffer from limited context length and noisy tool responses. A natural solution is to adopt a multi-agent framework with planner- and worker-agents to manage context. However, no existing methods support effective reinforcement learning post-training of tool-integrated multi-agent frameworks. To address this gap, we propose Multi-Agent Tool-Integrated Policy Optimization (MATPO), which enables distinct roles (planner and worker) to be trained within a single LLM instance using role-specific prompts via reinforcement learning. MATPO is derived from a principled credit assignment mechanism across planner and worker rollouts. This design eliminates the need to deploy multiple LLMs, which would be memory-intensive, while preserving the benefits of specialization. Experiments on GAIA-text, WebWalkerQA, and FRAMES show that MATPO consistently outperforms single-agent baselines by an average of 18.38% relative improvement in performance and exhibits greater robustness to noisy tool outputs. Our findings highlight the effectiveness of unifying multiple agent roles within a single LLM and provide practical insights for stable and efficient multi-agent RL training.

## Summary

This paper investigates multilingual routing patterns in Mixture-of-Experts (MoE) large language models and introduces intervention methods to improve cross-lingual performance. The key contributions are:

**Key Findings from Routing Analysis:**
The authors analyze four prominent MoE LLMs (Qwen3, Phi-3.5-MoE, GPT-OSS, OLMoE) using parallel multilingual datasets and discover that MoE models exhibit a U-shaped routing divergence pattern - early and late layers show language-specific routing, while middle layers demonstrate significant cross-lingual routing alignment. Crucially, they find a strong correlation between a model's performance in a language and how similarly its tokens are routed to English in these middle layers. The analysis also reveals that routing entropy decreases across layers more rapidly for non-English languages, and there's higher routing consistency within non-English sequences.

**Intervention Methodology:**
Building on these observations, the authors develop inference-time interventions that steer router logits to activate task-specific experts (identified using English domain data) during multilingual evaluation. They explore both soft interventions (adding/subtracting values proportional to logit standard deviation) and hard interventions (force-activation/deactivation). The interventions specifically target middle layers where language-universal representations are concentrated.

**Results:**
The interventions yield consistent improvements across three models (Qwen3, Phi-3.5-MoE, GPT-OSS) and two evaluation tasks (MGSM mathematical reasoning and Global-MMLU medical subset). Gains of 1-2% are observed across 15+ languages, with particularly strong improvements for lower-resource languages. The effectiveness is highly sensitive to target layers - interventions outside the identified middle layers or targeting multilingual-specialized experts instead of task experts lead to performance degradation.

The work demonstrates that improved cross-lingual routing alignment causally enhances multilingual generalization, revealing the modular separation between language-specific and language-universal parameters in MoE architectures and motivating future methods to promote expert sharing across languages.

## Critique

Of course. Here is a critique of the paper "Multilingual Routing in Mixture-of-Experts," focusing on its strengths, weaknesses, novelty, significance, and clarity.

### Overall Summary

This is a strong, well-executed paper that provides valuable insights into the inner workings of multilingual Mixture-of-Experts (MoE) models. The work successfully bridges the gap between interpretability research on dense models and the increasingly dominant MoE architecture, presenting compelling evidence for language-universal processing in middle layers and demonstrating a causal link between routing alignment and performance through effective interventions.

---

### Strengths

1.  **High Novelty and Timeliness:** The focus on interpreting multilingual behavior in MoE models is highly novel. While the "U-shape" finding (language-specific early/late layers, language-universal middle layers) aligns with prior work on dense models, demonstrating this phenomenon through the discrete, interpretable lens of expert routing is a significant contribution. The intervention methodology is a creative and powerful way to move from correlation to causation.

2.  **Significant and Actionable Results:** The key finding—that manually steering routers in middle layers to activate task-specific experts (identified from English data) improves multilingual performance—is both significant and practical. Achieving consistent, statistically significant gains of 1-2% across three different state-of-the-art models and two distinct tasks is a robust result. It strongly suggests that suboptimal routing, not a fundamental lack of capability, limits multilingual generalization in these models.

3.  **Rigor and Thoroughness:** The paper is exceptionally thorough. The authors:
    *   Analyze four different MoE models (Qwen3, Phi-3.5-MoE, GPT-OSS, OLMoE), showing the generality of their findings.
    *   Use multiple, parallel datasets (FLoRes, MGSM, Global-MMLU) for both analysis and evaluation.
    *   Systematically explore the intervention space (layer sensitivity, expert type, intervention strength/type) before presenting the optimal configuration, which adds credibility to their claims.
    *   Provide extensive appendices with additional plots and methodological details.

4.  **Clear and Effective Presentation:** The paper is well-structured and easy to follow. The use of visualizations (e.g., Figure 1 & 2) effectively communicates the core "U-shape" finding and the correlation with performance. The writing is precise, and the methodology is described in sufficient detail for reproducibility.

---

### Weaknesses

1.  **Modest Performance Gains:** While statistically significant, the performance improvements from the interventions (1-2%) are modest in absolute terms. The paper rightly frames this as impressive given the simplicity of the intervention versus the scale of the models, but it tempers the immediate practical impact. The work is more of a compelling "proof-of-concept" than a ready-to-deploy technique.

2.  **Limited Exploration of Negative Results:** The paper clearly states that many intervention strategies (e.g., deactivation, intervening outside middle layers) led to performance degradation. While this reinforces their main hypothesis, a more detailed analysis of *why* these specific interventions fail could provide even deeper insights into the model's routing mechanics. For instance, what happens when you force-activate a multilingual expert in the middle layers?

3.  **English as the Sole Pivot:** The entire analysis and intervention methodology are centered on aligning non-English languages with English. The authors briefly mention that using other high-resource languages as the pivot flattens the trends but do not explore this further. A more multilingual-centric analysis (e.g., aligning Spanish with Chinese) could reveal more about the underlying "language-universal" space and whether it is truly universal or simply an "Anglocentric" one.

4.  **Potential Overstatement of Modularity:** Finding 5 states there is "no overlap" between multilingual and task experts. This is a very strong claim. While likely true for the specific thresholds (`τ ≥ 0.3`) and datasets used, it may not hold for more nuanced tasks or different thresholds. The claim of "complete separation" could be slightly tempered.

---

### Detailed Assessment

*   **Novelty of the Approach:** **High.** Translating established interpretability methods from dense models to the sparse, modular setting of MoEs is a novel and fruitful direction. The intervention technique, while building on concepts like activation steering, is creatively adapted to the unique routing mechanism of MoEs.

*   **Significance of the Results:** **High.** The results provide a causal explanation for cross-lingual transfer in MoEs: it is mediated by the activation of shared, task-relevant experts in the middle layers. This finding has significant implications for future model design and training, motivating techniques that explicitly encourage this kind of cross-lingual expert sharing.

*   **Clarity of the Presentation:** **Excellent.** The paper is a model of clear scientific writing. The logical flow from analysis to hypothesis to intervention is seamless. The figures are informative, the methodology is transparent, and the results are presented honestly and comprehensively.

### Conclusion

This is a high-quality paper that makes a substantial contribution to the fields of NLP interpretability and multilingual modeling. Its strengths in novelty, rigor, and clarity far outweigh its minor weaknesses. It opens up a promising research direction for improving multilingual performance in large language models by manipulating their internal routing dynamics, either during inference or as a guide for more effective training strategies.

---

# Mitigating Forgetting Between Supervised and Reinforcement Learning Yields Stronger Reasoners

Authors: Xiangchi Yuan, Xiang Chen, Tong Yu, Dachuan Shi, Can Jin, Wenke Lee, Saayan Mitra

Keywords: Reinforcement Learning, Supervised Fine-tuning, Catastrophic Forgetting, Reasoning, Parameter Freezing, Data Efficiency

Comments: None

Paper link: [http://arxiv.org/abs/2510.04454v1](http://arxiv.org/abs/2510.04454v1)

## Abstract

Large Language Models (LLMs) show strong reasoning abilities, often amplified by Chain-of-Thought (CoT) prompting and reinforcement learning (RL). Although RL algorithms can substantially improve reasoning, they struggle to expand reasoning boundaries because they learn from their own reasoning trajectories rather than acquiring external knowledge. Supervised fine-tuning (SFT) offers complementary benefits but typically requires large-scale data and risks overfitting. Recent attempts to combine SFT and RL face three main challenges: data inefficiency, algorithm-specific designs, and catastrophic forgetting. We propose a plug-and-play framework that dynamically integrates SFT into RL by selecting challenging examples for SFT. This approach reduces SFT data requirements and remains agnostic to the choice of RL or SFT algorithm. To mitigate catastrophic forgetting of RL-acquired skills during SFT, we select high-entropy tokens for loss calculation and freeze parameters identified as critical for RL. Our method achieves state-of-the-art (SoTA) reasoning performance using only 1.5% of the SFT data and 20.4% of the RL data used by prior SoTA, providing an efficient and plug-and-play solution for combining SFT and RL in reasoning post-training.

## Summary

Based on the provided paper, here is a summary focusing on its key contributions, methods, and results:

**Key Contributions:**
This paper introduces MIFO (Mitigating Forgetting Between Supervised and Reinforcement Learning), a novel framework designed to effectively combine Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) for enhancing the reasoning capabilities of Large Language Models (LLMs). The authors identify and address a critical challenge in existing methods: catastrophic forgetting, where SFT's extensive parameter updates overwrite the knowledge acquired during RL training. They also tackle the issues of data inefficiency and algorithm-specific design dependencies common in prior SFT+RL approaches.

**Methods:**
MIFO is built on two core mechanisms to limit the disruptive impact of SFT on RL-learned knowledge:
1.  **Data Processing:** The framework dynamically interleaves RL and SFT phases. It uses RL rollouts to identify challenging questions (those with low accuracy) and populates an SFT buffer only with these cases and their verified solutions. Furthermore, during SFT, the loss is calculated only on high-entropy tokens (positions of high model uncertainty), which focuses learning and reduces unnecessary updates on already-confident tokens.
2.  **Parameter Freezing:** MIFO tracks which parameters are most updated (and thus deemed important) during an RL phase. It then freezes these RL-critical parameters during the subsequent SFT phase to protect them from being overwritten. These parameters are unfrozen for the next RL cycle. This design is informed by the key insight that SFT updates are "redundant" (some can be dropped without major performance loss) while RL updates are "parsimonious" (more critical and sensitive to change).

**Results:**
The proposed MIFO framework achieves state-of-the-art reasoning performance on multiple mathematical benchmarks (including AIME, AMC, OlympiadBench, and MATH500) using the Qwen2.5-Math models (1.5B and 7B parameters). Notably, it does so with significantly improved data efficiency, requiring only **1.5% of the SFT data** and **20.4% of the RL data** compared to the previous SOTA method. Additionally, models trained with MIFO generate more concise reasoning traces (shorter response lengths) while maintaining or improving accuracy, demonstrating enhanced reasoning efficiency. Ablation studies confirm that both the entropy-based token selection and parameter freezing components are crucial for MIFO's performance gains.

## Critique

Of course. Here is a critique of the paper "Mitigating Forgetting Between Supervised and Reinforcement Learning Yields Stronger Reasoners," focusing on its strengths and weaknesses.

### Overall Summary

This paper presents **MIFO**, a novel framework designed to effectively combine Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) for improving the reasoning capabilities of Large Language Models (LLMs). The core contribution is the identification and mitigation of "catastrophic forgetting," where SFT overwrites knowledge acquired during RL. The results are significant, demonstrating state-of-the-art performance with dramatically improved data and response-length efficiency.

---

### Strengths

1.  **Novel and Well-Motivated Problem Formulation:** The paper's central thesis—that SFT causes catastrophic forgetting of RL-gained skills due to its larger, more redundant parameter updates—is a fresh and compelling insight. It moves beyond simply combining SFT and RL to diagnosing a fundamental flaw in how they interact. The initial experiments in Section 3 (gradient dropping, update magnitude) are elegant and provide strong, empirical motivation for the proposed method.

2.  **Innovative and Cohesive Method Design:** MIFO is not a single trick but a cohesive framework with two synergistic components that directly address the identified problem:
    *   **Data Processing:** Selecting only challenging examples for SFT and focusing the loss on high-entropy tokens is a clever way to reduce the overall magnitude and "blast radius" of SFT updates.
    *   **Parameter Freezing:** Dynamically identifying and freezing parameters critical to RL performance is a direct and intuitive solution to the core problem of SFT overwriting. The use of a historical importance map (`C_i`) is a sophisticated touch that accounts for long-term training stability.

3.  **Significant and Impressive Results:** The experimental results are a major strength. Achieving state-of-the-art performance while using only **1.5% of the SFT data and 20.4% of the RL data** of the previous best method is a remarkable claim that, if reproducible, represents a substantial leap in training efficiency. The simultaneous improvement in **response length efficiency** (shorter, more concise reasoning chains) is another highly valuable and practical outcome.

4.  **Rigorous and Extensive Evaluation:** The paper is thorough in its evaluation, using multiple model sizes (1.5B and 7B), a wide array of strong baselines, and a comprehensive suite of reasoning benchmarks. The inclusion of ablations, hyperparameter analyses, and template robustness studies in the main paper and appendix adds considerable weight to the claims.

---

### Weaknesses

1.  **Clarity of Presentation and Writing:**
    *   The writing, while technically sound, is occasionally dense and could benefit from clearer phrasing. Some sentences are long and complex, which can hinder readability.
    *   The use of acronyms is slightly excessive. While "MIFO" is fine, introducing `MIFO†` for the `α=0` variant so early can be confusing. A simple phrase like "MIFO (no history)" in the table would be clearer for a first-time reader.
    *   The connection between certain design choices and their observed effects is sometimes left for the reader to infer. For example, the hypothesis for *why* Parameter Freezing leads to shorter responses (Section 5.4) is buried in the caption of Figure 7 and could be more explicitly stated in the main text.

2.  **Limited Exploration of the "Why":** While the paper excellently demonstrates *that* SFT updates are more redundant and RL's are more parsimonious, it offers only a brief theoretical appendix (Appendix C) on the underlying cause. A more in-depth discussion or analysis of *why* these two training paradigms exhibit such fundamentally different update characteristics would strengthen the foundational contribution. Is it the nature of the loss functions? The data distribution?

3.  **Narrow Problem Scope:** The evaluation is exclusively on mathematical reasoning tasks. While this is a standard and challenging domain, the paper's claims are framed more generally ("reasoning," "LLMs"). It remains an open question whether the SFT-RL forgetting phenomenon and MIFO's effectiveness generalize to other reasoning domains (e.g., commonsense, symbolic) or tasks like instruction-following and dialogue. A discussion of this limitation would be appropriate.

4.  **Complexity and Computational Overhead:** The method introduces non-trivial complexity: maintaining a rolling buffer of SFT data, computing token-wise entropies for the entire SFT dataset, and tracking/updating a parameter importance map. The paper does not discuss the computational overhead of these steps compared to simpler baselines like `SFT->RL` or `ReLIFT`. This is a relevant practical consideration for adoption.

---

### Conclusion

This is a **high-quality, impactful paper** that identifies a genuine and previously underexplored problem in LLM post-training. The proposed MIFO framework is novel, well-designed, and directly tackles the problem with an elegant two-pronged approach. The empirical results are strong and convincingly demonstrate substantial improvements in both performance and efficiency.

The main weaknesses lie in the presentation, which could be more accessible, and the scope of the evaluation, which is currently limited to mathematical reasoning. Nonetheless, the core insights and the effectiveness of the method are likely to influence future work on combining diverse learning signals in LLM training.

