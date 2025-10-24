---
title: "ArXiv Daily Digest on 2025-10-23"
collection: digests
permalink: /digests/arxiv_cs_CL_2025-10-23_report
date: 2025-10-23
location: "Online"
---

Today's research landscape showcases a compelling focus on enhancing large reasoning models (LRMs) through tool integration, multilingual reasoning, and multi-agent collaboration. Several papers explore how LRMs can be trained to effectively use external tools like Code Interpreters (CIs), with frameworks such as CoRT (Code-Optimized Reasoning Training) demonstrating significant gains in mathematical reasoning efficiency. In multilingual contexts, studies reveal that while English often serves as a "reasoning lingua franca," this approach risks "Lost in Translation" errors, highlighting the need for robust native-language reasoning capabilities. Meanwhile, multi-agent reinforcement learning systems like Mixture-of-Minds are achieving state-of-the-art results in complex tasks like table understanding by decomposing problems into specialized planning, coding, and answering roles. Another emerging trend is data efficiency, with methods like LM-Mixup showing that low-quality data, when properly distilled via instruction distillation, can rival the performance of training on full datasets. Finally, new evaluation techniques such as ThinMQM (Thinking-calibrated Multidimensional Quality Metrics) are calibrating LRM "thinking" to improve machine translation assessment, reducing computational budgets while boosting accuracy.

## TL;DR

Total papers: 64 , Selected papers: 5

Here's a TL;DR summary of the key themes and insights from these papers:

**Main Theme: Advancing Reasoning and Efficiency in Language Models**

These papers collectively explore methods to enhance language models' reasoning capabilities while improving efficiency through specialized training techniques and multi-agent frameworks.

**Key Insights:**

1. **Tool-Integrated Reasoning** - CoRT (https://arxiv.org/abs/2510.20342) demonstrates that strategically injecting hints (Hint-Engineering) with just 30 high-quality samples can teach models to effectively use code interpreters, reducing token usage by 30-50% while improving mathematical reasoning.

2. **Multilingual Reasoning Trade-offs** - (https://arxiv.org/abs/2510.20647) reveals that while English reasoning generally yields better performance, it introduces "Lost in Translation" errors, highlighting the need for balanced multilingual capabilities.

3. **Multi-Agent Specialization** - Mixture-of-Minds (https://arxiv.org/abs/2510.20176) shows that decomposing table reasoning into planning, coding, and answering agents with RL optimization can surpass proprietary models like OpenAI-o4-mini-high, achieving 62.13% on TableBench.

4. **Data Efficiency through Distillation** - LM-Mixup (https://arxiv.org/abs/2510.20449) introduces Instruction Distillation, transforming low-quality data into valuable training resources, enabling models trained on just 3% of data to match full-dataset performance.

5. **Calibrated Evaluation** - ThinMQM (https://arxiv.org/abs/2510.20780) addresses LRM overthinking in MT evaluation, reducing thinking budgets by 35x while improving correlation with human judgments by +8.7 points.

**Common Trend**: The papers emphasize quality over quantity - whether in data (small, high-quality samples), reasoning (calibrated thinking), or specialization (multi-agent roles) - to achieve better performance with improved efficiency.

---

# Teaching Language Models to Reason with Tools

Authors: Chengpeng Li, Zhengyang Tang, Ziniu Li, Mingfeng Xue, Keqin Bao, Tian Ding, Ruoyu Sun, Benyou Wang, Xiang Wang, Junyang Lin, Dayiheng Liu

Keywords: Tool-integrated reasoning, Code interpreters, Mathematical reasoning, Reinforcement learning, Hint-engineering, Token efficiency, Large reasoning models

Comments: NIPS2025 Accepted

Paper link: [http://arxiv.org/abs/2510.20342v1](http://arxiv.org/abs/2510.20342v1)

## Abstract

Large reasoning models (LRMs) like OpenAI-o1 have shown impressive capabilities in natural language reasoning. However, these models frequently demonstrate inefficiencies or inaccuracies when tackling complex mathematical operations. While integrating computational tools such as Code Interpreters (CIs) offers a promising solution, it introduces a critical challenge: a conflict between the model's internal, probabilistic reasoning and the external, deterministic knowledge provided by the CI, which often leads models to unproductive deliberation. To overcome this, we introduce CoRT (Code-Optimized Reasoning Training), a post-training framework designed to teach LRMs to effectively utilize CIs. We propose \emph{Hint-Engineering}, a new data synthesis strategy that strategically injects diverse hints at optimal points within reasoning paths. This approach generates high-quality, code-integrated reasoning data specifically tailored to optimize LRM-CI interaction. Using this method, we have synthesized 30 high-quality samples to post-train models ranging from 1.5B to 32B parameters through supervised fine-tuning. CoRT further refines the multi-round interleaving of external CI usage and internal thinking by employing rejection sampling and reinforcement learning. Our experimental evaluations demonstrate CoRT's effectiveness, yielding absolute improvements of 4\% and 8\% on DeepSeek-R1-Distill-Qwen-32B and DeepSeek-R1-Distill-Qwen-1.5B, respectively, across five challenging mathematical reasoning datasets. Moreover, CoRT significantly enhances efficiency, reducing token usage by approximately 30\% for the 32B model and 50\% for the 1.5B model compared to pure natural language reasoning baselines. The models and code are available at: https://github.com/ChengpengLi1003/CoRT.

## Summary

Of course. Here is a summary of the paper "Teaching Language Models to Reason with Tools," focusing on its key contributions, methods, and results.

This paper introduces CoRT (Code-Optimized Reasoning Training), a framework designed to teach Large Reasoning Models (LRMs) to effectively and efficiently use Code Interpreters (CIs) for complex mathematical reasoning. The core challenge addressed is the conflict between a model's internal, probabilistic reasoning and the external, deterministic knowledge from a CI, which often leads to inefficiencies like delayed code computation and unnecessary verification of code results.

The key methodological contribution is a novel data synthesis strategy called **Hint-Engineering**. Instead of relying on large-scale automated data generation, this approach involves strategically inserting targeted hints into reasoning paths at optimal points to teach the model when and how to use the CI. For instance, hints like "It looks tedious, we can use python code" are inserted when manual calculation begins, and "We don’t need to doubt the accuracy of python calculations" are used to prevent redundant verification. Remarkably, the authors created a high-quality dataset of only **30 manually verified samples** using this method. The CoRT framework then uses this data for Supervised Fine-Tuning (SFT) and Rejection Fine-Tuning (RFT) on larger models (32B parameters). For smaller models (1.5B), they employ a strong-to-weak distillation followed by **Code-Integrated Reinforcement Learning (RL)** with a custom reward that combines answer accuracy and code execution success.

The results demonstrate the effectiveness of CoRT:
*   **Performance Gains:** The Hint-Engineering approach achieved significant absolute improvements, with a **4%** gain for a 32B model and an **8%** gain for a 1.5B model across five challenging mathematical benchmarks (AIME24, AIME25, AMC23, MATH500, OlympiadBench).
*   **Superior Efficiency:** A major success of Hint-Engineering is its dramatic improvement in token efficiency. It reduced token consumption by approximately **30%** for the 32B model and **50%** for the 1.5B model compared to baseline methods, while maintaining competitive accuracy.
*   **Improved Code Behavior:** Analysis showed that the Hint-Engineering model learned a more balanced and effective use of code, shifting from primarily using code for verification (as in a baseline "Prompt-Hint" method) to using it more frequently for direct calculation.

In conclusion, this work shows that high-quality, strategically engineered data is more impactful than large quantities of data for teaching tool use. By combining Hint-Engineering with a tailored training pipeline, CoRT successfully enhances both the performance and efficiency of LRMs in mathematical reasoning tasks.

## Critique

Based on the provided paper "Teaching Language Models to Reason with Tools" (CoRT), here is a balanced assessment of its strengths and weaknesses.

### Strengths

1.  **Novelty of the Core Approach (Hint-Engineering):** The paper introduces "Hint-Engineering," a novel and targeted data synthesis strategy. This is a key contribution, moving beyond simple prompting ("Prompt-Hint") to strategically inject human-designed hints at specific points in the reasoning process to correct identified inefficiencies like "delayed code computation" and "code result distrust." This human-in-the-loop approach for high-quality data creation is a significant conceptual advance.

2.  **Significance of Results (Efficiency):** The paper demonstrates highly significant and practical results, particularly in **token efficiency**. Achieving a 30-50% reduction in token usage for the 32B and 1.5B models, respectively, while maintaining or improving performance is a major finding. This directly addresses cost and latency concerns in deploying large reasoning models.

3.  **Comprehensive and Scalable Framework (CoRT):** The proposed CoRT framework is well-structured and comprehensive. It systematically explores a pipeline from data synthesis (Prompt-Hint, Hint-Engineering) to training (SFT, RFT) and further refinement (Strong-to-Weak Distillation, RL), demonstrating its applicability across different model sizes (1.5B to 32B). The successful application of RL to tool-integrated reasoning for 1.5B models is a notable technical contribution.

4.  **Rigorous and Multi-faceted Evaluation:** The evaluation is thorough, using five challenging mathematical benchmarks. The analysis goes beyond mere accuracy to include crucial aspects like token efficiency, Pass@k analysis, and a detailed qualitative analysis of code behavior (calculation vs. verification), which provides deep insights into *how* the models' reasoning strategies change.

5.  **Clarity of Presentation:** The paper is generally well-written and logically structured. The problem formulation is clear, the methodology is explained step-by-step, and the figures (especially Figures 1, 3, and 5) effectively illustrate the core concepts, comparisons, and results.

### Weaknesses

1.  **Scalability of the Hint-Engineering Method:** The most significant weakness is the **extremely small scale of the manually crafted Hint-Engineering dataset (only 30 samples)**. While the authors convincingly argue for "less is more" and the power of high-quality data, this raises immediate questions about the generalizability and scalability of this method. Manually crafting optimal hints for thousands of diverse problems is not a feasible long-term strategy. The paper would be stronger if it proposed or explored a path towards automating or scaling this hint-engineering process.

2.  **Limited RL on Larger Models:** The authors explicitly state that applying RL to the 32B model was computationally infeasible. This leaves an open question: would RL provide the same performance boost to the larger, more capable models as it did for the 1.5B models? The results are therefore incomplete for the frontier-scale models discussed.

3.  **Incremental Nature of Performance Gains:** While the efficiency gains are dramatic, the raw performance (accuracy) improvements, though consistent, are more modest (e.g., +4% for 32B, +8% for 1.5B). The 32B models do not clearly surpass the best-existing tool-integrated baselines like STILL-3-TOOL-32B in overall accuracy, though they achieve these results much more efficiently. The claim of a "performance breakthrough" in the abstract might be slightly overstated when considering accuracy alone.

4.  **Clarity on the RL Reward Function:** The design of the code execution reward `R_c` is somewhat simplistic (a penalty of -1 only if *all* code executions fail). It is unclear why this specific formulation was chosen over a more nuanced reward that might, for example, penalize individual failed executions or reward successful ones. A more detailed ablation or justification for this design choice would strengthen the methodology section.

### Overall Assessment

This is a strong paper with a highly novel core idea (Hint-Engineering) that leads to impactful and well-demonstrated results, particularly in the crucial area of reasoning efficiency. The work is timely, addressing a key challenge in integrating deterministic tools with probabilistic LLM reasoning. The main weaknesses revolve around the scalability of its most innovative data synthesis method and the incomplete exploration of RL across all model scales. Despite these limitations, the paper makes a valuable contribution by providing a clear framework and compelling evidence for the power of small, high-quality datasets and strategic training to optimize tool use in large reasoning models.

---

# The Reasoning Lingua Franca: A Double-Edged Sword for Multilingual AI

Authors: Alan Saji, Raj Dabre, Anoop Kunchukuttan, Ratish Puduppully

Keywords: multilingual reasoning, large reasoning models, cognitive behaviors, lost in translation, cross-lingual transfer

Comments: 14 pages, 13 figures, 5 tables

Paper link: [http://arxiv.org/abs/2510.20647v1](http://arxiv.org/abs/2510.20647v1)

## Abstract

Large Reasoning Models (LRMs) achieve strong performance on mathematical, scientific, and other question-answering tasks, but their multilingual reasoning abilities remain underexplored. When presented with non-English questions, LRMs often default to reasoning in English, raising concerns about interpretability and the handling of linguistic and cultural nuances. We systematically compare an LRM's reasoning in English versus the language of the question. Our evaluation spans two tasks: MGSM and GPQA Diamond. Beyond measuring answer accuracy, we also analyze cognitive attributes in the reasoning traces. We find that English reasoning traces exhibit a substantially higher presence of these cognitive behaviors, and that reasoning in English generally yields higher final-answer accuracy, with the performance gap increasing as tasks become more complex. However, this English-centric strategy is susceptible to a key failure mode - getting "Lost in Translation," where translation steps lead to errors that would have been avoided by question's language reasoning.

## Summary

Here is a summary of the paper "The Reasoning Lingua Franca: A Double-Edged Sword for Multilingual AI":

**Key Contributions:**
This paper provides a systematic analysis of how Large Reasoning Models (LRMs) perform when reasoning in English versus the question's original language for multilingual tasks. The study highlights a fundamental tension in multilingual AI: while English reasoning generally yields better performance, it introduces translation errors that can be avoided by reasoning in the native language.

**Methods:**
The authors evaluated LRMs (Qwen3-32B and DeepSeek-R1-Distill-Llama-70B) on two benchmark datasets with varying difficulty levels: MGSM (mathematical reasoning) and GPQA Diamond (expert-level scientific questions). They compared three key aspects: (1) final answer accuracy, (2) cognitive behaviors in reasoning traces (subgoal setting, verification, backtracking, backward chaining), and (3) "Lost in Translation" errors where translation to English introduces mistakes. The study covered multiple languages grouped by resource availability (high-resource to low-resource) and used GPT-4o-mini as an evaluator for cognitive behavior analysis.

**Key Results:**
1. **English Reasoning Superiority**: Models consistently achieved higher final-answer accuracy when reasoning in English, with the performance gap widening for lower-resource languages and more complex tasks (GPQA Diamond vs MGSM).

2. **Cognitive Behavior Disparity**: English reasoning traces exhibited substantially richer cognitive behaviors (subgoal setting, verification, backtracking, backward chaining) compared to native-language reasoning.

3. **Lost in Translation Vulnerability**: A significant portion of English reasoning errors (up to 77% for low-resource languages in MGSM) stemmed from translation mistakes that wouldn't occur in native-language reasoning, revealing a critical weakness in the English-centric approach.

The study concludes that while English reasoning currently dominates performance, developing robust native-language reasoning capabilities is essential for truly multilingual AI systems that can handle linguistic and cultural nuances without translation-induced errors.

## Critique

Of course. Here is a critique of the strengths and weaknesses of the paper "The Reasoning Lingua Franca: A Double-Edged Sword for Multilingual AI".

### Overall Summary

This is a well-structured and timely empirical study that provides a clear, data-driven analysis of a critical issue in multilingual AI: the trade-offs between using English as a reasoning lingua franca versus reasoning in a question's native language. The paper's strength lies in its systematic multi-faceted evaluation, but it is primarily a diagnostic study rather than one that proposes a novel solution.

---

### Strengths

1.  **Clear and Important Research Question:** The paper tackles a highly relevant and underexplored problem. As Large Reasoning Models (LRMs) become more prevalent, understanding the implications of their English-centric reasoning bias is crucial for global and equitable AI deployment. The "double-edged sword" framing is effective and accurately reflects the core findings.

2.  **Comprehensive and Multi-faceted Evaluation:** The authors go beyond simply measuring final-answer accuracy, which is a significant strength. The analysis is structured around three insightful perspectives:
    *   **Final Answer Accuracy:** The standard and most direct metric.
    *   **Cognitive Behaviors in Reasoning Traces:** This adds a valuable qualitative and mechanistic dimension, showing that English reasoning isn't just more accurate but also "richer" in terms of problem-solving strategies like backtracking and verification.
    *   **The "Lost in Translation" (LiT) Failure Mode:** This is a particularly compelling part of the paper. It successfully identifies and quantifies a critical weakness of the English-reasoning strategy, providing a concrete counter-argument to simply always using English.

3.  **Robust Experimental Design:** The use of two datasets with varying difficulty (MGSM vs. GPQA Diamond) effectively shows that the English-reasoning advantage becomes more pronounced with task complexity. The inclusion of languages with different resource levels (high, mid, low) adds nuance to the findings. The use of sampling with multiple runs and reporting standard deviations is methodologically sound.

4.  **Clear and Effective Presentation:** The paper is well-written and the findings are communicated clearly through a combination of text and well-designed visualizations. The heatmaps (Figures 1 & 3) and bar charts (Figures 2, 4, & 5) effectively summarize complex, multi-dimensional results.

---

### Weaknesses and Limitations

1.  **Limited Novelty in Approach:** The core methodology is largely an application of existing techniques. The use of system prompts and prefix tokens to steer the reasoning language follows directly from cited prior work (e.g., Yong et al., 2025). The study excels in its *comparative analysis* rather than proposing a new architectural or training method to *solve* the identified problem.

2.  **Diagnostic, Not Prescriptive:** The paper thoroughly diagnoses the problem but offers little in the way of a path forward. While the conclusion calls for "targeted efforts in dataset construction, training objectives, and evaluation," it does not propose or experiment with any specific techniques to improve native-language reasoning. This leaves it as a strong position paper that highlights an area for future work rather than a technical advancement itself.

3.  **Potential Critiques of the "Cognitive Behavior" Analysis:** While innovative, the method of using `gpt-4o-mini` to classify cognitive behaviors introduces a dependency on another LLM's judgment. Although the authors mention manual verification on a subset, the scalability and potential biases of this automated evaluation are a minor weakness. A more rigorous inter-annotator agreement or validation would strengthen this section.

4.  **Acknowledged but Significant Limitations:** The authors correctly note important limitations:
    *   **Narrow Task Domain:** The findings are confined to mathematical and scientific reasoning (MGSM, GPQA). The conclusions may not hold for tasks requiring deep cultural understanding, creative writing, or legal reasoning, where nuances in the native language are paramount.
    *   **Model Scope:** The study is limited to open-weight models. The behavior of state-of-the-art proprietary reasoning models (like OpenAI's o1 series) might differ significantly and is a critical unknown.

5.  **Clarity of Contribution vs. Related Work:** While Section 2 and Appendix A mention related work, the paper could do a slightly better job of crisply articulating its specific advancements over studies like Yong et al. (2025) or Wang et al. (2025). The extension to more complex tasks (GPQA), the cognitive behavior analysis, and the quantitative LiT metric are the key differentiators, and this could be emphasized more.

---

### Conclusion

This is a valuable and solid piece of research that makes a clear contribution to the field. Its significance lies in its empirical rigor and comprehensive diagnosis of a pressing issue. It convincingly demonstrates that while English is currently the most effective language for complex reasoning in LRMs, this strategy has a fundamental flaw ("Lost in Translation") that prevents it from being a robust, universal solution. The main weakness is its lack of a proposed remedy, positioning it as an excellent foundation and call to action for future research rather than a standalone technical breakthrough.

---

# Mixture-of-Minds: Multi-Agent Reinforcement Learning for Table Understanding

Authors: Yuhang Zhou, Mingrui Zhang, Ke Li, Mingyi Wang, Qiao Liu, Qifei wang, Jiayi Liu, Fei Liu, Serena Li, Weiwi Li, Mingze Gao, Abhishek Kumar, Xiangjun Fan, Zhuokai Zhao, Lizhu Zhang

Keywords: Multi-Agent Reinforcement Learning, Table Understanding, Mixture-of-Minds, Monte Carlo Tree Search, Group Relative Policy Optimization, Test-Time Scaling

Comments: 18 pages, 4 figures

Paper link: [http://arxiv.org/abs/2510.20176v1](http://arxiv.org/abs/2510.20176v1)

## Abstract

Understanding and reasoning over tables is a critical capability for many real-world applications. Large language models (LLMs) have shown promise on this task, but current approaches remain limited. Fine-tuning based methods strengthen language reasoning; yet they are prone to arithmetic errors and hallucination. In contrast, tool-based methods enable precise table manipulation but rely on rigid schemas and lack semantic understanding. These complementary drawbacks highlight the need for approaches that integrate robust reasoning with reliable table processing. In this work, we propose Mixture-of-Minds, a multi-agent framework that decomposes table reasoning into three specialized roles: planning, coding, and answering. This design enables each agent to focus on a specific aspect of the task while leveraging code execution for precise table manipulation. Building on this workflow, we introduce a self-improvement training framework that employs Monte Carlo Tree Search (MCTS) rollouts to generate pseudo-gold trajectories and optimize agents with reinforcement learning (RL). Extensive experiments show that Mixture-of-Minds delivers substantial gains, reaching 62.13% on TableBench and surpassing OpenAI-o4-mini-high. These results demonstrate the promise of combining structured multi-agent workflows with RL to advance table understanding.

## Summary

Here is a concise summary of the paper "Mixture-of-Minds: Multi-Agent Reinforcement Learning for Table Understanding":

**Key Contributions:**
This paper introduces Mixture-of-Minds, a novel multi-agent framework that decomposes table reasoning into three specialized roles: planning, coding, and answering agents. The key innovation is combining this structured workflow with a self-improvement training framework that uses Monte Carlo Tree Search (MCTS) rollouts to generate pseudo-gold intermediate supervision and optimizes agents using Group Relative Policy Optimization (GRPO).

**Methods:**
The framework employs three specialized agents: (1) a planning agent that outlines reasoning steps, (2) a coding agent that generates and executes Python code for precise table manipulation, and (3) an answering agent that synthesizes the final answer. To address the challenge of lacking intermediate supervision, the authors use MCTS-style rollouts to explore multiple reasoning paths and extract high-quality plans and codes from successful trajectories. Each agent is then optimized sequentially using GRPO with tailored reward functions that capture plan quality, code execution validity, and final answer accuracy.

**Results:**
Extensive experiments on TableBench and FinQA datasets demonstrate that Mixture-of-Minds achieves state-of-the-art performance, reaching 62.13% accuracy on TableBench with Qwen3-32B - surpassing proprietary models like OpenAI o4-mini-high (61.69%). The framework shows consistent improvements across model sizes (8B-70B) and tasks, with particular strength in numerical reasoning. Test-time scaling further boosts performance, and ablation studies confirm that sequential training of all three agents provides incremental gains, with the answering agent training yielding the most significant improvements in data analysis tasks.

## Critique

Of course. Here is a critique of the paper "Mixture-of-Minds: Multi-Agent Reinforcement Learning for Table Understanding," focusing on its strengths and weaknesses.

### Strengths

1.  **High Novelty in Approach:** The core contribution is highly novel and addresses a clear gap in the literature. The integration of a specialized multi-agent workflow (Planning, Coding, Answering) with a self-improving, MCTS-driven training framework is a sophisticated and ambitious solution. It successfully bridges the gap between purely model-based reasoning (prone to errors) and rigid tool-based methods (lacking semantics). The use of MCTS to generate pseudo-gold supervision for intermediate steps is a clever and practical solution to a fundamental problem in training multi-step reasoning systems.

2.  **Significant and Compelling Results:** The results are a major strength of the paper. The authors demonstrate that their method enables smaller open-source models (e.g., Qwen3-32B) to not only outperform their own baselines by a large margin but also to **surpass state-of-the-art proprietary models like OpenAI's o4-mini-high** (62.13% vs. 61.69%). This is a powerful claim that, if valid, signifies a substantial advancement in the field. The consistent improvements across multiple model families and sizes (8B to 70B) strongly validate the general applicability of their framework.

3.  **Comprehensive Experimental Design:** The evaluation is thorough. The authors test on both in-domain (TableBench) and out-of-domain (FinQA) datasets, showing the method's robustness. They compare against a wide array of strong baselines, including proprietary models, alternative training methods (GRPO, DAPO), and prior SOTA table-specific models (Table-R1, Table-LLM). The inclusion of test-time scaling (TTS) analysis further strengthens the paper by showing how the modular agent design can effectively leverage additional inference-time compute.

4.  **Clear and Well-Structured Presentation:** The paper is exceptionally well-written and structured. The problem is clearly motivated, the workflow is explained with a helpful diagram (Fig. 1), and the complex training procedure is broken down into digestible steps (MCTS data generation, followed by sequential GRPO training for each agent). The ablation study and TTS analysis provide clear insights into the contribution of each component.

### Weaknesses

1.  **Complexity and Computational Cost:** The primary weakness is the immense computational overhead. The MCTS-style data generation requires sampling a large number of trajectories (α × β × γ), and the sequential GRPO training of three separate agents is highly resource-intensive. While the results are impressive, the practical barrier to replicating or building upon this work is very high. A discussion of the computational budget (e.g., GPU hours) would have been valuable for context.

2.  **Ablation on MCTS Necessity:** While the sequential training ablation is insightful, a more critical ablation is missing: **How crucial is the MCTS component versus simply using the best-of-N samples from a base model?** The authors assume that the intermediate steps from a successful trajectory are "pseudo-gold," but it's unclear how much the exploration and selective retention of MCTS contributes over a simpler, cheaper sampling-and-filtering approach.

3.  **Limited Analysis of Failure Modes:** The paper convincingly shows *that* the method works, but could do more to explain *how* and *when* it fails. A qualitative analysis of error cases would be enlightening. For instance, does the system typically fail due to poor plans, non-executable code, or the answering agent misinterpreting correct evidence? Understanding the bottlenecks would guide future improvements.

4.  **Reward Function Design:** The reward functions for the planning and coding agents (e.g., using BLEU against a "pseudo-gold" sample) are pragmatic but potentially limiting. BLEU may not perfectly capture plan quality or code semantic equivalence. The weights in the composite rewards (e.g., `0.1*r_fmt + 0.9*r_BLEU`) also seem somewhat arbitrary. A more principled analysis or ablation of the reward design would strengthen the methodology.

### Summary

This is a strong paper that presents a novel, effective, and well-evaluated framework for a challenging problem. The key strength is its demonstration that a carefully designed multi-agent system with a sophisticated training recipe can allow smaller models to compete with and even exceed the performance of much larger, state-of-the-art proprietary models. The main weaknesses revolve around its high computational complexity and a lack of deeper analysis into the necessity of its most expensive component (MCTS) and its specific failure modes. Despite these points, the significance of the results and the novelty of the approach make it a notable contribution.

---

# LM-mixup: Text Data Augmentation via Language Model based Mixup

Authors: Zhijie Deng, Zhouan Shen, Ling Li, Yao Zhou, Zhaowei Zhu, Yanji He, Wei Wang, Jiaheng Wei

Keywords: Instruction Distillation, Data Augmentation, Reinforcement Learning, Low-Quality Data, Language Model Mixup, GRPO, Multi-Dimensional Rewards

Comments: None

Paper link: [http://arxiv.org/abs/2510.20449v1](http://arxiv.org/abs/2510.20449v1)

## Abstract

Instruction tuning is crucial for aligning Large Language Models (LLMs), yet the quality of instruction-following data varies significantly. While high-quality data is paramount, it is often scarce; conversely, abundant low-quality data is frequently discarded, leading to substantial information loss. Existing data augmentation methods struggle to augment this low-quality data effectively, and the evaluation of such techniques remains poorly defined. To address this, we formally define the task of Instruction Distillation: distilling multiple low-quality and redundant inputs into high-quality and coherent instruction-output pairs. Specifically, we introduce a comprehensive data construction pipeline to create MIXTURE, a 144K-sample dataset pairing low-quality or semantically redundant imperfect instruction clusters with their high-quality distillations. We then introduce LM-Mixup, by first performing supervised fine-tuning on MIXTURE and then optimizing it with reinforcement learning. This process uses three complementary reward signals: quality, semantic alignment, and format compliance, via Group Relative Policy Optimization (GRPO). We demonstrate that LM-Mixup effectively augments imperfect datasets: fine-tuning LLMs on its distilled data, which accounts for only about 3% of the entire dataset, not only surpasses full-dataset training but also competes with state-of-the-art high-quality data selection methods across multiple benchmarks. Our work establishes that low-quality data is a valuable resource when properly distilled and augmented with LM-Mixup, significantly enhancing the efficiency and performance of instruction-tuned LLMs.

## Summary

Here is a summary of the paper "LM-mixup: Text Data Augmentation via Language Model based Mixup":

**Key Contributions:**
The paper introduces a new task called *Instruction Distillation*, which aims to transform multiple low-quality, redundant, or incomplete instruction-following samples into a single high-quality, information-dense output. To support this paradigm, the authors construct Mixture, a 144K-sample dataset with hierarchical mappings from 2-20 low-quality variants to high-quality targets across five task types. They also propose LM-Mixup, a training framework that effectively distills imperfect data into valuable training resources.

**Methods:**
The LM-Mixup framework involves two main stages. First, cold-start pretraining on the Mixture dataset provides the model with basic generation and fusion capabilities. Then, Group Relative Policy Optimization (GRPO) reinforcement learning is applied with three complementary reward signals: quality (using a KNN-Bayes scoring scheme), semantic alignment (via embedding similarity), and format compliance. This multi-dimensional reward design ensures the model produces outputs that are not only high-quality but also faithful to the input semantics and structurally correct.

**Results:**
Experiments demonstrate that LM-Mixup significantly outperforms supervised fine-tuning and strong baselines on the Mixture test set. More importantly, when applied to downstream tasks, training on distilled data (combining original high-quality samples with LM-Mixup processed low-quality data) using only 10K samples (≈3% of the full dataset) matches or surpasses full-dataset training and state-of-the-art data selection methods on the OpenLLM Leaderboard. The 50% mixup + 50% original configuration achieves the best average performance across multiple benchmarks, showing that properly distilled low-quality data can become a valuable resource for enhancing model training efficiency and performance.

## Critique

Of course. Here is a commentary on the strengths and weaknesses of the paper "LM-mixup: Text Data Augmentation via Language Model based Mixup".

### Summary

This paper introduces "Instruction Distillation," a task focused on transforming multiple low-quality or redundant text inputs into a single, high-quality output. To support this, the authors construct the "Mixture" dataset and develop "LM-Mixup," a model trained via supervised fine-tuning (SFT) followed by reinforcement learning (RL) using Group Relative Policy Optimization (GRPO) with multi-dimensional rewards. The core finding is that a model fine-tuned on a small subset (≈3%) of data—comprising original high-quality samples and samples distilled from low-quality data—can match or surpass the performance of models trained on the full, large-scale dataset and state-of-the-art data selection methods.

---

### Strengths

1.  **High Novelty and Well-Defined Task:** The paper formalizes the "Instruction Distillation" task, which is a novel and impactful formulation. Instead of simply discarding low-quality data or using basic augmentation, it proposes a method to actively *synthesize* and *enhance* it. This addresses a significant, real-world problem in data-centric AI and has the potential to drastically improve data efficiency.

2.  **Comprehensive and Rigorous Methodology:** The approach is thorough and well-engineered.
    *   **Dataset Construction:** The creation of the "Mixture" dataset is a major contribution. The pipeline for generating high- and low-quality variants, including cross-topic fusion and noise injection, is detailed and designed to build a robust model.
    *   **Multi-Faceted Training Pipeline:** The combination of cold-start SFT followed by GRPO-based RL is a robust training strategy. The use of three distinct, complementary rewards (Quality, Semantic Alignment, Format Compliance) effectively guides the model to produce outputs that are not just high-quality but also faithful to the input and structurally sound.

3.  **Significant and Compelling Results:** The experimental results are a key strength of the paper.
    *   The most striking finding is that using only 3.3% of the data (a mix of original high-quality and distilled data) can outperform training on the full 300K-sample dataset. This demonstrates exceptional data efficiency.
    *   The method competes with or surpasses several strong baselines (e.g., DS2, DEITA) on the OpenLLM leaderboard, validating its effectiveness against established data selection techniques.
    *   The ablation studies are convincing, clearly showing the necessity of each reward component and the scalability of the approach to larger models.

4.  **Clear Presentation and High Reproducibility:** The paper is generally well-structured and easy to follow. The figures effectively illustrate the core concepts of the task and the training pipeline. The authors have committed to releasing both the code and the "Mixture" dataset, which is crucial for reproducibility and will likely foster further research in this area.

---

### Weaknesses

1.  **Potential Bias in Evaluation:** A notable weakness is the reliance on "ChatGPT-4o-mini" for both constructing the dataset (quality scoring) and evaluating the model's performance on the Mixture test set. This creates a potential for circular reasoning or bias, where the judge model has preferences that align with the data it helped create. While the authors briefly acknowledge this in Section 5.3 and test with the DS2 pipeline, a more robust evaluation involving human assessment or a suite of diverse judge models would have strengthened the results.

2.  **Limited Analysis of "Distilled" Data Characteristics:** The paper convincingly shows that the distilled data works, but it provides less insight into *why* it works so well. A deeper analysis of the distilled samples—e.g., quantifying their complexity, diversity, or novelty compared to the original high-quality data—would offer more profound insights into the mechanism behind the performance gains.

3.  **Clarity on the "Mixup" Terminology:** The term "Mixup" is borrowed from computer vision, where it typically refers to the linear interpolation of inputs or features. While the paper uses it metaphorically for "mixing" information from multiple inputs, this could cause confusion. A more precise term like "instruction fusion" or "data consolidation" might have been clearer, or a more detailed justification for keeping the "Mixup" name could have been provided.

4.  **Computational Cost of the Pipeline:** The full pipeline—involving dataset creation with a powerful LLM, cold-start SFT, and RL fine-tuning—is likely computationally expensive. While the final result is data-efficient for the downstream task, the cost of training the LM-Mixup model itself is not discussed. An analysis of the training costs would provide a more complete picture of the method's practical utility.

---

### Overall Assessment

This is a strong paper that makes a valuable contribution to the field of data-efficient LLM training. The novelty of the "Instruction Distillation" task, the rigor of the proposed "LM-Mixup" method, and the compelling empirical results demonstrating significant data compression are highly impressive. Despite some weaknesses, primarily concerning evaluation bias and terminology, the core ideas and findings are significant. The work successfully demonstrates that low-quality data is an undervalued resource that, when properly distilled, can be a powerful asset for enhancing LLM performance.

---

# Are Large Reasoning Models Good Translation Evaluators? Analysis and Performance Boost

Authors: Runzhe Zhan, Zhihong Huang, Xinyi Yang, Lidia S. Chao, Min Yang, Derek F. Wong

Keywords: Large Reasoning Models, Machine Translation Evaluation, MQM Framework, Thinking Calibration, ThinMQM, Evaluation Metrics

Comments: NeurIPS 2025

Paper link: [http://arxiv.org/abs/2510.20780v1](http://arxiv.org/abs/2510.20780v1)

## Abstract

Recent advancements in large reasoning models (LRMs) have introduced an intermediate "thinking" process prior to generating final answers, improving their reasoning capabilities on complex downstream tasks. However, the potential of LRMs as evaluators for machine translation (MT) quality remains underexplored. We provides the first systematic analysis of LRM-as-a-judge in MT evaluation. We identify key challenges, revealing LRMs require tailored evaluation materials, tend to "overthink" simpler instances and have issues with scoring mechanisms leading to overestimation. To address these, we propose to calibrate LRM thinking by training them on synthetic, human-like thinking trajectories. Our experiments on WMT24 Metrics benchmarks demonstrate that this approach largely reduces thinking budgets by ~35x while concurrently improving evaluation performance across different LRM scales from 7B to 32B (e.g., R1-Distill-Qwen-7B achieves a +8.7 correlation point improvement). These findings highlight the potential of efficiently calibrated LRMs to advance fine-grained automatic MT evaluation.

## Summary

This paper presents the first systematic investigation into using Large Reasoning Models (LRMs) as evaluators for machine translation quality assessment. The authors analyze LRM performance under the MQM (Multidimensional Quality Metrics) framework and identify several key challenges: LRMs require tailored evaluation materials (reference-free for strong models, reference-based for weaker ones), tend to "overthink" simpler instances, and suffer from scoring mechanism issues leading to overestimation biases.

The main contribution is **ThinMQM (Thinking-calibrated MQM)**, a method that calibrates LRM thinking by training them on synthetic, human-like evaluation trajectories. This approach generates structured thinking chains that mimic the human two-phase MQM evaluation process: error span annotation followed by rubric-based scoring. The authors fine-tune various LRMs (7B to 32B parameters) on this synthetic data.

Results on WMT24 Metrics benchmarks show significant improvements: **ThinMQM reduces thinking budgets by ~35x** while improving evaluation performance across all model scales. Notably, R1-Distill-Qwen-7B achieves a **+8.7 correlation point improvement**. The method achieves performance comparable to state-of-the-art metrics like xCOMET while being more efficient. Analysis reveals that ThinMQM primarily works by calibrating scoring distributions and reducing the overestimation problem, making model predictions more aligned with human judgments. The approach also demonstrates good generalization to low-resource language pairs and robustness to prompt variations.

## Critique

Of course. Here is a detailed critique of the paper "Are Large Reasoning Models Good Translation Evaluators? Analysis and Performance Boost," focusing on its strengths and weaknesses.

### **Overall Summary**

This is a strong, well-executed paper that makes a timely and valuable contribution to the fields of machine translation evaluation and reasoning models. It provides the first systematic analysis of Large Reasoning Models (LRMs) as evaluators, identifies key failure modes, and proposes a simple, effective, and well-validated solution. The work is characterized by thorough experimentation, clear presentation, and significant, reproducible results.

---

### **Strengths**

1.  **High Novelty and Timeliness:** The paper addresses a clear gap in the literature. While "LLM-as-a-judge" is a well-established paradigm, the exploration of "LRM-as-a-judge" (with explicit reasoning chains) is novel, especially in the context of the complex, multi-step MQM evaluation framework. The research question is highly relevant given the rapid emergence of models like DeepSeek-R1.

2.  **Rigorous and Comprehensive Analysis:** The paper's strength lies in its systematic deconstruction of the problem. The analysis is not superficial; it delves into critical, often overlooked aspects:
    *   **Material Contribution (Section 3.2):** The use of Shapley values to quantify the contribution of source vs. reference text is a sophisticated and insightful approach. The finding that optimal materials depend on model scale is a crucial, non-obvious insight for practitioners.
    *   **Scoring Mechanism Pitfalls (Section 3.3):** The investigation into rule-based vs. model-based scoring and the associated "attribution dilemma" is important. It effectively argues for the transparency and robustness of rule-based scoring in this context.
    *   **Thinking Budget Analysis (Section 3.4):** The concept of "overthinking" is well-supported by the data, showing that LRMs are inefficient and do not rationally allocate computational effort based on instance difficulty. The comparison with base LLMs is also telling.

3.  **Effective and Practical Solution (ThinMQM):** The proposed method is elegant and directly addresses the identified problems.
    *   **Simplicity:** The core idea—synthesizing human-aligned thinking trajectories from existing MQM data and fine-tuning the LRM on them—is straightforward and does not require complex new architectures or massive new datasets.
    *   **Significant Results:** The performance improvements are substantial (e.g., +8.7 points for a 7B model) and bring smaller models to a performance level competitive with state-of-the-art metrics like xCOMET. The ~35x reduction in thinking budget is a major practical achievement, making LRM evaluation vastly more efficient.

4.  **Excellent Experimental Design and Evaluation:**
    *   The use of the latest WMT24 benchmark helps avoid data contamination concerns.
    *   The evaluation is comprehensive, covering multiple model sizes (7B to 671B), language pairs, and both system- and segment-level metrics.
    *   The inclusion of significance testing and extensive ablation studies (stability, generalization, prompt sensitivity) greatly strengthens the validity of the claims.

5.  **Clarity of Presentation:** The paper is very well-structured and easy to follow. The research framework in Figure 2 effectively guides the reader through the analytical and proposed sections. The writing is clear, and the figures and tables are well-designed and support the narrative effectively.

---

### **Weaknesses**

1.  **Limited Model Diversity:** The primary analysis is heavily reliant on the DeepSeek-R1 family of models (and its distillations). While justified by their transparent reasoning process, it raises the question of whether the findings (e.g., the scale-dependent preference for source/reference) are general properties of LRMs or specific to the R1 architecture and training method. Including another open-source LRM family, if available, would have strengthened the generalizability of the conclusions.

2.  **Ablation on Data Scale and Quality:** While the ThinMQM method is proven effective, there is limited ablation on the synthetic data itself. How does the performance scale with the amount of synthetic data? How sensitive is the method to the quality and diversity of the human MQM data used for synthesis? A brief analysis here could provide valuable guidance for future applications.

3.  **Superficial Treatment of "Why ThinMQM Works":** The paper clearly shows *that* ThinMQM works, primarily by calibrating the score distribution. However, the analysis of *why* the calibrated thinking process leads to better scores is somewhat surface-level. A deeper qualitative analysis of the "thinking" generated by ThinMQM versus the baseline—comparing the logic, error identification, and potential reasoning errors—could provide more profound insights into the mechanics of improved evaluation.

4.  **Minor Presentation Issue:** The reference to "Figure˜1" and "Table˜2" in the text (with a tilde) appears to be a formatting artifact from LaTeX that was not fully cleaned up in the conversion to this markdown format. This is a minor issue but slightly detracts from the polish.

---

### **Conclusion**

This is a high-quality paper that makes a significant contribution. Its strengths in novelty, analytical depth, practical impact, and experimental rigor far outweigh its minor weaknesses. The proposed ThinMQM method is a simple yet powerful technique that effectively harnesses the potential of LRMs for MT evaluation while mitigating their inefficiencies and biases. The work is likely to influence both future research in automated evaluation and the practical deployment of LRM-based metrics.

