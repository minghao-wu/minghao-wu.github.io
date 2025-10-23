---
title: "ArXiv Daily Digest on 2025-10-22"
collection: digests
permalink: /digests/arxiv_cs_CL_2025-10-22_report
date: 2025-10-22
location: "Online"
---

Today's research landscape showcases significant advancements in enhancing large language model (LLM) reasoning through sophisticated reinforcement learning (RL) and multi-agent frameworks. A prominent trend is addressing the "learning cliff" phenomenon, where models plateau on problems beyond their current capabilities, with novel solutions like Scaf-GRPO (Scaffolded Group Relative Policy Optimization) providing hierarchical hints to restore learning signals. Concurrently, AgenticMath demonstrates the power of multi-agent systems for generating high-quality, mathematically rigorous synthetic data, proving that targeted data quality can outperform massive-scale alternatives. The scope of agent capabilities continues to expand, as evidenced by ColorAgent's robust, personalized operating system agent, while multilingual performance sees a paradigm shift with prompt-space optimization—systematically transforming prompts for naturalness, cultural adaptation, and difficulty rather than merely translating them. Furthermore, LoongRL tackles advanced long-context reasoning by synthesizing high-difficulty tasks that induce emergent "plan–retrieve–reason–recheck" patterns, enabling impressive generalization from shorter training contexts.

## TL;DR

Total papers: 71 , Selected papers: 5

**TL;DR: Recent arXiv Papers on RL for LLMs, Data Generation, and Multilinguality**

This batch of papers focuses on enhancing LLM capabilities through reinforcement learning, synthetic data generation, and multilingual optimization. Key themes include overcoming learning plateaus in reasoning tasks, improving data quality over quantity, and addressing cultural biases in multilingual models.

**Key Papers:**

1. **Scaf-GRPO** (https://arxiv.org/abs/2510.19807v1): Introduces scaffolded RL with hierarchical hints to overcome the "learning cliff" where models fail on problems beyond their current capabilities. Achieves 44.3% improvement on math benchmarks by providing minimal guidance only when learning plateaus.

2. **AgenticMath** (https://arxiv.org/abs/2510.19361v1): Proposes multi-agent framework for generating high-quality math QA pairs. Demonstrates that targeted, quality data (30-60K samples) outperforms large-scale alternatives (400K-2.3M samples), emphasizing data quality over quantity.

3. **LoongRL** (https://arxiv.org/abs/2510.19363v1): Addresses long-context reasoning through UUID chain-based data synthesis. Trains on 16K contexts but generalizes to 128K, inducing emergent "plan-retrieve-reason-recheck" patterns that rival frontier models.

4. **ColorAgent** (https://arxiv.org/abs/2510.19386v1): Presents comprehensive OS agent framework combining step-wise RL with personalized interaction, achieving SOTA on mobile automation benchmarks while transitioning agents from tools to "warm partners."

5. **The Art of Asking** (https://arxiv.org/abs/2510.19806v1): Shifts focus from completion to prompt optimization for multilingual data, using naturalness, cultural adaptation, and difficulty enhancements to reduce English-centric biases and improve performance across 12 languages.

**Main Insights:**
- RL methods are evolving beyond simple reward optimization to include scaffolding and structured reasoning patterns
- High-quality, targeted synthetic data generation is more efficient than large-scale, low-quality alternatives
- Multilingual performance benefits from cultural adaptation and prompt optimization rather than just translation
- Agent frameworks are becoming more sophisticated, combining environmental interaction with personalized user alignment

---

# Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning

Authors: Xichen Zhang, Sitong Wu, Yinghao Zhu, Haoru Tan, Shaozuo Yu, Ziyi He, Jiaya Jia

Keywords: Scaf-GRPO, Reinforcement Learning, LLM Reasoning, Learning Cliff, Hierarchical Hints, Policy Optimization, Mathematics Benchmarks

Comments: Code: https://github.com/dvlab-research/Scaf-GRPO

Paper link: [http://arxiv.org/abs/2510.19807v1](http://arxiv.org/abs/2510.19807v1)

## Abstract

Reinforcement learning from verifiable rewards has emerged as a powerful technique for enhancing the complex reasoning abilities of Large Language Models (LLMs). However, these methods are fundamentally constrained by the ''learning cliff'' phenomenon: when faced with problems far beyond their current capabilities, models consistently fail, yielding a persistent zero-reward signal. In policy optimization algorithms like GRPO, this collapses the advantage calculation to zero, rendering these difficult problems invisible to the learning gradient and stalling progress. To overcome this, we introduce Scaf-GRPO (Scaffolded Group Relative Policy Optimization), a progressive training framework that strategically provides minimal guidance only when a model's independent learning has plateaued. The framework first diagnoses learning stagnation and then intervenes by injecting tiered in-prompt hints, ranging from abstract concepts to concrete steps, enabling the model to construct a valid solution by itself. Extensive experiments on challenging mathematics benchmarks demonstrate Scaf-GRPO's effectiveness, boosting the pass@1 score of the Qwen2.5-Math-7B model on the AIME24 benchmark by a relative 44.3% over a vanilla GRPO baseline. This result demonstrates our framework provides a robust and effective methodology for unlocking a model's ability to solve problems previously beyond its reach, a critical step towards extending the frontier of autonomous reasoning in LLM.

## Summary

Based on the provided paper, here is a summary of "Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning":

**Key Problem and Contribution:** This paper addresses the "learning cliff" phenomenon in Reinforcement Learning from Verifier Rewards (RLVR), where language models fail to learn from problems beyond their current capabilities because all attempts yield zero rewards, collapsing the learning gradient. The authors introduce Scaf-GRPO, a novel training framework that applies pedagogical scaffolding principles to provide minimal, hierarchical guidance only when models reach these learning plateaus, enabling them to conquer previously intractable problems.

**Methodology:** Scaf-GRPO operates in two phases: (1) a guidance exemption period that allows models to independently solve "pseudo-hard" problems through exploration, and (2) hierarchical hint-guided exploration for "true-hard" problems. The framework uses a three-tiered hint system (knowledge → planning → solution) injected directly into prompts, proceeding from abstract to concrete guidance until the model succeeds. Crucially, it maintains on-policy integrity by replacing failed trajectories with successful guided ones in the same batch, restoring the learning signal without altering the fundamental GRPO optimization objective.

**Key Results:** The method demonstrates significant improvements across multiple challenging math benchmarks. On Qwen2.5-Math-7B, Scaf-GRPO achieved a 44.3% relative improvement over vanilla GRPO on AIME24 and outperformed strong baselines including LUFFY (9.2% gain). The framework showed broad applicability across different model architectures (Qwen, Llama), scales (1.5B-7B), and specializations (including LongCoT models). Ablation studies confirmed the importance of progressive guidance, the complete hint hierarchy, and incremental hint delivery. The method also demonstrated strong out-of-distribution generalization on scientific reasoning tasks.

## Critique

Of course. Here is a critique of the paper "Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning," covering its strengths, weaknesses, and overall presentation.

### Strengths

1.  **Novel and Well-Motivated Approach:** The paper identifies a clear and significant problem in RL for LLMs—the "learning cliff"—where models plateau because they consistently fail on a subset of problems, leading to zero reward signals and vanishing gradients. The proposed solution, "scaffolding" through in-prompt hints, is a clever and intuitive application of educational theory to machine learning. It is a distinct and novel alternative to the prevalent prefix-continuation methods (like LUFFY), as it avoids distributional mismatch and preserves the model's exploratory autonomy.

2.  **Comprehensive and Convincing Experimental Setup:** The authors conduct extensive experiments across a diverse set of models (different architectures, sizes, and specializations) and benchmarks. This thoroughness strongly supports their claim that Scaf-GRPO is a robust and model-agnostic framework. The inclusion of a LongCoT model and out-of-distribution evaluation (GPQA) further strengthens the case for the method's generalization capabilities.

3.  **Significant and Quantifiable Results:** The performance improvements are substantial and clearly presented. A **44.3% relative improvement** on the challenging AIME24 benchmark (Qwen2.5-Math-7B) over the vanilla GRPO baseline is a standout result. Consistently outperforming strong baselines like LUFFY, Oat-Zero, and Simple-RL demonstrates that the approach is not just an incremental improvement but a meaningful advancement.

4.  **Rigorous Ablation Studies:** The paper includes a detailed ablation study that systematically validates key design choices:
    *   The necessity of the **guidance exemption period**.
    *   The efficacy of the **progressive, hierarchical hint structure** (K→P→S).
    *   The importance of **incremental guidance**.
    These ablations are crucial for convincing the reader that every component of the proposed framework is well-justified and contributes to the final performance.

### Weaknesses

1.  **Practical Scalability and Dependency:** The most significant limitation, which the authors acknowledge, is the reliance on a pre-defined, high-quality, three-tiered hint hierarchy. Generating these hints for a large-scale, diverse dataset requires significant manual effort or a very capable "teacher" model (like DeepSeek-R1, as used here). This dependency could hinder the method's adoption and scalability compared to methods that might use simpler forms of guidance.

2.  **Computational Overhead:** The process of "hierarchical hint-guided exploration" involves multiple sequential generations for a single problem until a correct solution is found (querying the model with `H_knowledge`, then `H_planning`, then potentially chunks of `H_solution`). This iterative process is computationally more expensive per problem than standard GRPO or even single-shot prefix methods, which could be a practical constraint.

3.  **Limited Exploration of Hint Quality Impact:** While the method depends on high-quality hints, there is no analysis of how robust it is to noisy or suboptimal hints. An ablation studying the impact of hint quality (e.g., using a weaker model to generate them) would have been valuable to understand the method's sensitivity and practical boundaries.

4.  **Narrow Domain Focus:** The evaluation is heavily focused on mathematical reasoning. While this is a valid and challenging domain, the paper's conclusion that it fosters "fundamental" reasoning skills would be stronger if supported by results on a wider range of tasks, such as logical deduction, code generation, or commonsense reasoning. The single OOD benchmark (GPQA) is a good start but not sufficient to claim broad generality.

### Clarity and Presentation

*   **Clarity:** The paper is generally well-written and clearly structured. The "learning cliff" analogy is effective, and Figure 3 provides an excellent high-level overview of the framework. The methodology is described in sufficient detail, with appendices offering further formalization and prompt examples.
*   **Presentation:** The use of tables, figures, and ablations is effective and supports the narrative. The results are presented transparently, showing improvements and occasional regressions. The distinction between "true-hard" and "pseudo-hard" problems is a useful conceptual tool.

### Overall Summary

This is a **strong paper** that makes a meaningful contribution to the field of RL for LLM reasoning. Its core strength lies in its novel and well-executed approach to a known, critical problem. The results are significant, empirically robust, and clearly demonstrate the superiority of the proposed method over existing baselines. The primary weaknesses relate to practical scalability (hint generation cost) and a somewhat narrow domain of evaluation, but these do not undermine the core contribution. The paper is likely to influence future work on guided reinforcement learning and curriculum design for LLMs.

---

# AgenticMath: Enhancing LLM Reasoning via Agentic-based Math Data Generation

Authors: Xianyang Liu, Yilin Liu, Shuai Wang, Hao Cheng, Andrew Estornell, Yuzhi Zhao, Jiaheng Wei

Keywords: multi-agent systems, mathematical reasoning, synthetic data generation, large language models, data quality, chain-of-thought reasoning

Comments: Work in progress

Paper link: [http://arxiv.org/abs/2510.19361v1](http://arxiv.org/abs/2510.19361v1)

## Abstract

The creation of high-quality datasets to improve Large Language Model (LLM) reasoning remains a significant challenge, as current methods often suffer from generating low-quality/incorrect answers and limited information richness from available data sources. To address this, we propose AgenticMath, a novel agentic pipeline for generating high-quality mathematical question-answer pairs to enhance the supervised fine-tuning of LLMs. Our method operates through four stages: (1) Seed Question Filter that selects questions with high information richness, complexity, and clarity; (2) an Agentic Question Rephrase step that employs a multi-agent system to generate diverse, logically consistent paraphrases; (3) an Answer Augment step where rewrite answers using chain-of-thought reasoning to enhance numerical and logical correctness, without reliance on human-provided labels; and (4) a final Question and Answer Evaluation that retains only the most superior pairs. Extensive experiments demonstrate that, fine-tuning 3B-8B parameter LLMs on AgenticMath generated datasets (comprising only 30-60K math samples) achieves competitive or superior performance on diverse in domain and out-of-domain mathematical reasoning benchmarks compared to baselines trained on much more data (e.g., 400K or 2.3M samples). Our work demonstrates that targeted, high-quality data generation is a more efficient path to improving mathematical reasoning in LLMs than large-scale, low-quality alternatives.

## Summary

Based on the provided paper, here is a summary of "AgenticMath: Enhancing LLM Reasoning via Agentic-based Math Data Generation":

**Key Contributions:**
The paper introduces AgenticMath, a novel multi-agent framework for generating high-quality mathematical question-answer pairs to enhance LLM reasoning. The main contributions include: (1) the AgenticMath framework itself, which provides a systematic paradigm for building high-quality reasoning corpora; (2) AgenticMathQA, a curated dataset in 30K, 60K, and 90K versions that emphasizes clarity, correctness, and diversity over scale; and (3) comprehensive empirical validation showing superior data efficiency compared to existing methods.

**Methods:**
AgenticMath operates through four coordinated stages: (1) Seed Problem Filtering that selects high-value problems based on complexity, information value, and clarity using LLM-based scoring and curation; (2) Agentic Problem Generation involving three specialized agents (Rephrase, Review, Revise) that iteratively refine problems through paraphrasing and quality control; (3) Solution Generation where a solver agent produces detailed chain-of-thought reasoning paths; and (4) Synthetic Data Evaluation that applies multi-dimensional scoring and diversity-based selection to retain only top-quality problem-solution pairs. The framework uses GPT-4o-mini for all agent operations and eliminates reliance on human annotations.

**Results:**
The method achieves state-of-the-art performance with remarkable data efficiency. When fine-tuning 3B-8B parameter LLMs on only 30K-60K samples, AgenticMath matches or surpasses baselines trained on 400K-2.3M samples across six mathematical reasoning benchmarks. For example, AgenticMath-Qwen2.5-3B with 30K samples achieves 53.7 average accuracy, outperforming MathFusion by over 15 points. The results demonstrate that AgenticMath delivers competitive or superior performance with 5-15% of the data size used by conventional methods, highlighting that targeted high-quality data generation is more effective than large-scale, low-quality alternatives for improving mathematical reasoning in LLMs.

## Critique

Of course. Here is a critique of the paper "AgenticMath: Enhancing LLM Reasoning via Agentic-based Math Data Generation."

### Strengths

1.  **Strong Empirical Results and Clear Contribution:** The paper's most compelling strength is its robust empirical evidence. The results convincingly demonstrate that the proposed method, using only 30K-60K samples, can match or even surpass models trained on datasets 6-70 times larger (400K-2.3M samples). This directly supports the central thesis that "data quality trumps quantity," a significant and valuable finding for the field, especially given the high cost of large-scale data generation and training.

2.  **Well-Structured and Systematic Pipeline:** The proposed 4-stage pipeline (Seed Filtering, Agentic Problem Generation, Solution Generation, and Synthetic Data Evaluation) is logical, well-motivated, and clearly explained. The integration of a multi-agent system specifically for the nuanced task of *problem* generation and refinement, rather than just solution generation, is a notable strength. The iterative "Review-Revise" loop adds a layer of rigor often missing in synthetic data creation.

3.  **Comprehensive Evaluation:** The evaluation is thorough, testing on a diverse set of six benchmarks covering both in-domain (GSM8K, MATH) and out-of-domain tasks (e.g., CollegeMath, OlympiadBench). Testing across multiple base models (Qwen, DeepSeekMath, Mistral, Llama) strengthens the claim of the method's generalizability and is a best practice in the field.

4.  **Insightful Ablation Studies:** The ablation study in Section 4.3 is a key strength. It systematically breaks down the contribution of each pipeline stage, providing evidence that each component contributes to the final performance. This moves beyond a "black box" presentation and offers valuable insights into *why* the method works.

### Weaknesses

1.  **Limited Novelty in Core Concepts:** While the *orchestration* is novel and effective, the individual components are largely built upon established ideas. LLM-based filtering/scoring (e.g., DS²), multi-agent systems for data generation (e.g., AgentInstruct), and Chain-of-Thought solution generation are well-known techniques. The paper's primary novelty lies in the careful integration and specialization of these concepts for the specific, high-stakes domain of mathematical reasoning, rather than in a fundamentally new algorithmic breakthrough.

2.  **High Dependency on a Powerful, Proprietary LLM:** The entire AgenticMath pipeline relies exclusively on GPT-4o-mini for all agent roles (scoring, rephrasing, reviewing, solving). This raises concerns about cost, reproducibility, and generalizability. It is unclear if the impressive results are due to the clever pipeline design or simply the inherent capability of the underlying, powerful model. The paper would be significantly strengthened by an analysis using open-source models for the agent roles to demonstrate the method's independence from a specific, closed-source API.

3.  **Opaque Cost and Computational Overhead:** The paper does not discuss the computational cost or latency of running the entire multi-agent, multi-stage pipeline. Generating 30K high-quality samples likely requires a massive number of API calls (for scoring, generating multiple variants, iterative review-revise loops, etc.). A discussion of the trade-off between data efficiency at training time and the overhead at data generation time is a notable omission.

4.  **Ambiguity in Final Dataset Composition:** The description of how the final 30K/60K/90K datasets are constructed is slightly confusing. For the 30K setting, it's stated as 15K seed + 15K synthetic. For 60K/90K, it's described as duplicating the 30K set. This duplication strategy is unconventional and could simply be a data augmentation technique rather than generating truly new, diverse samples. A clearer explanation of the dataset scaling strategy and its implications would be beneficial.

### Clarity of Presentation

The paper is generally well-written and clearly structured. The use of figures for each stage of the pipeline is excellent and aids comprehension. The tables are comprehensive, though dense; highlighting the key comparisons (as done with the blue rows) is helpful.

A minor point for improvement: The "Problem Definition" section (3.1) uses a standard SFT loss function, which is arguably unnecessary for this audience and could be stated more succinctly without the formula.

### Overall Significance

Despite its weaknesses, the paper makes a significant contribution. It provides a compelling, empirically-validated blueprint for generating high-quality data efficiently. The "less is more" philosophy, backed by strong results, is an important counterpoint to the prevailing trend of massive dataset scaling. Researchers and practitioners looking to improve model performance in reasoning-heavy domains without access to immense computational resources will find this work highly relevant and inspiring. It successfully shifts the focus from "how much data" to "how good is the data."

---

# ColorAgent: Building A Robust, Personalized, and Interactive OS Agent

Authors: Ning Li, Qiqiang Lin, Zheng Wu, Xiaoyun Mo, Weiming Zhang, Yin Zhao, Xiangmou Qu, Jiamu Zhou, Jun Wang, Congmin Zheng, Yuanyi Song, Hongjiang Chen, Heyuan Huang, Jihong Wang, Jiaxin Yin, Jingwei Yu, Junwei Liao, Qiuying Peng, Xingyu Lou, Jun Wang, Weiwen Liu, Zhuosheng Zhang, Weinan Zhang

Keywords: OS Agent, Multi-agent Framework, Reinforcement Learning, Personalized Interaction, Mobile GUI Automation, Human-Agent Interaction, Self-evolving Training

Comments: None

Paper link: [http://arxiv.org/abs/2510.19386v1](http://arxiv.org/abs/2510.19386v1)

## Abstract

With the advancements in hardware, software, and large language model technologies, the interaction between humans and operating systems has evolved from the command-line interface to the rapidly emerging AI agent interactions. Building an operating system (OS) agent capable of executing user instructions and faithfully following user desires is becoming a reality. In this technical report, we present ColorAgent, an OS agent designed to engage in long-horizon, robust interactions with the environment while also enabling personalized and proactive user interaction. To enable long-horizon interactions with the environment, we enhance the model's capabilities through step-wise reinforcement learning and self-evolving training, while also developing a tailored multi-agent framework that ensures generality, consistency, and robustness. In terms of user interaction, we explore personalized user intent recognition and proactive engagement, positioning the OS agent not merely as an automation tool but as a warm, collaborative partner. We evaluate ColorAgent on the AndroidWorld and AndroidLab benchmarks, achieving success rates of 77.2% and 50.7%, respectively, establishing a new state of the art. Nonetheless, we note that current benchmarks are insufficient for a comprehensive evaluation of OS agents and propose further exploring directions in future work, particularly in the areas of evaluation paradigms, agent collaboration, and security. Our code is available at https://github.com/MadeAgents/mobile-use.

## Summary

Here is a summary of the paper "ColorAgent: Building A Robust, Personalized, and Interactive OS Agent":

**Key Contributions:**
ColorAgent presents a comprehensive framework for developing operating system (OS) agents that can engage in long-horizon, robust interactions with mobile environments while enabling personalized and proactive user interaction. The work makes two main contributions: (1) a tailored training paradigm and multi-agent framework for robust environmental interaction, and (2) mechanisms for personalized and proactive user interaction to transform agents from cold task executors into warm collaborative partners.

**Methods:**
The system employs a two-stage training paradigm. Stage I uses step-wise reinforcement learning with Group Relative Policy Optimization (GRPO) to optimize single-step decision-making, incorporating multi-path augmentation and difficulty-based filtering to handle GUI complexity. Stage II implements self-evolving training through an iterative cycle of trajectory rollout, filtering, and fine-tuning to scale data generation automatically. For the agent framework, ColorAgent adopts a multi-agent architecture with three specialized modules: knowledge retrieval for generalization, task orchestration for memory transfer and consistency, and hierarchical reflection (action, trajectory, and global levels) for error recovery. Additionally, the paper explores personalized user intent recognition (with user memory) and proactive engagement (without user memory) to better align with human intentions.

**Results:**
ColorAgent achieves state-of-the-art performance on AndroidWorld (77.2% success rate) and AndroidLab (50.7% success rate), significantly outperforming existing proprietary models, open models, and frameworks. Ablation studies confirm the complementary benefits of both the training strategies and agent-level modules. The personalized and proactive interaction modules also demonstrate strong performance on specialized benchmarks MobileIAR (58.66% intention alignment rate) and VeriOS-Bench (68.98% success rate), showing the system's ability to function as a warm partner rather than just a utilitarian tool.

## Critique

Of course. Here is a critique of the paper "ColorAgent: Building A Robust, Personalized, and Interactive OS Agent," focusing on its strengths and weaknesses.

### **Strengths**

1.  **Comprehensive and Ambitious Scope:** The paper tackles the full-stack challenge of building an Operating System (OS) agent, addressing not only robust task execution (environment interaction) but also the more nuanced problem of user alignment (human interaction). This dual focus on being both a capable "tool" and a "warm partner" is a significant and valuable contribution that goes beyond most existing work in the field.

2.  **Well-Structured, Multi-Faceted Approach:** The proposed solution is not a single trick but a cohesive system built on several well-motivated pillars. The two-stage training paradigm (Step-wise RL + Self-Evolving) effectively targets core model capabilities, while the multi-agent framework (Knowledge Retrieval, Task Orchestration, Hierarchical Reflection) directly addresses the identified failure modes of a single agent. The clear linkage between the diagnosed problems (Section 3.1) and the proposed solutions (Section 3.2) strengthens the paper's narrative.

3.  **Strong Empirical Results:** The paper demonstrates state-of-the-art (SOTA) performance on established and challenging benchmarks like AndroidWorld and AndroidLab. The ablation studies are particularly compelling, showing that each component of both the training and the framework contributes to the final performance. This provides strong evidence for the effectiveness of the proposed methods.

4.  **Practicality and Focus on Real-World Challenges:** The authors show a keen awareness of real-world deployment issues. The "multi-path augmentation" during data construction acknowledges that there is often more than one correct way to complete a task, a nuance often missed in benchmarks. The discussion on security, collaboration penalties, and the limitations of current benchmarks (Section 6) is thoughtful and positions the work as a step towards a practical system, not just a benchmark optimizer.

### **Weaknesses**

1.  **Limited Novelty in Individual Components:** While the overall system is novel and impactful, many of the core techniques are adaptations or compositions of existing ideas. Step-wise RL, self-evolving/self-improvement loops, RAG, task decomposition, and reflection mechanisms are all active areas of research. The paper's primary novelty lies in the specific integration and application of these techniques to the complex problem of mobile OS agents, rather than in inventing fundamentally new algorithms.

2.  **Under-Specified "Warm Partner" Components:** Section 5, which covers the transition from "tool to partner," feels less developed than the core environment interaction parts. The modules for personalized intent recognition and proactive engagement are described at a high level and referenced via other papers. While the results on MobileIAR and VeriOS-Bench are strong, a more detailed explanation within this paper of how these modules are integrated into ColorAgent would have been beneficial. This section currently reads more like a summary of plug-in capabilities rather than a core part of the presented system.

3.  **Clarity and Presentation Hiccups:**
    *   **Repetitive Structure:** The paper's structure is very clear but becomes somewhat repetitive. For example, the contributions listed in the introduction are essentially re-stated as the section headings for the core methodology (Sections 2 and 3).
    *   **Inconsistent Model Naming:** In Table 1, the final "ColorAgent" result is listed under the "Frameworks" section as an evolution of "GUI-Owl-32B + Model Training + Agent Framework." It would be clearer to have a dedicated row for "ColorAgent (Ours)" to immediately highlight the final system's performance against other frameworks and models.
    *   **Figure Quality:** The figures (e.g., Figure 1, 2, 4) are functional but somewhat simplistic and could be more informative. For instance, Figure 4 could better illustrate the data flow between the different agents in the framework.

### **Overall Assessment**

This is a strong technical report that presents a comprehensive and high-performing system for mobile OS agents. The significance of the work lies in its holistic approach, successfully combining advanced training techniques with a sophisticated multi-agent framework to achieve robust task execution. The results are impressive and convincingly demonstrate the value of the proposed architecture.

The main weaknesses are a reliance on synthesizing established techniques rather than introducing radical new ones, and a somewhat superficial treatment of the "warm partner" aspects compared to the deep dive on the "tool" aspects. The presentation is generally clear but could be polished for greater impact.

In summary, **ColorAgent represents a significant engineering and research milestone in building practical, general-purpose OS agents, setting a new state-of-the-art and providing a valuable blueprint for future work in the field.**

---

# The Art of Asking: Multilingual Prompt Optimization for Synthetic Data

Authors: David Mora, Viraat Aryabumi, Wei-Yin Ko, Sara Hooker, Julia Kreutzer, Marzieh Fadaee

Keywords: multilingual prompt optimization, synthetic data generation, cultural adaptation, difficulty enhancement, naturalness transformation, LLM fine-tuning

Comments: None

Paper link: [http://arxiv.org/abs/2510.19806v1](http://arxiv.org/abs/2510.19806v1)

## Abstract

Synthetic data has become a cornerstone for scaling large language models, yet its multilingual use remains bottlenecked by translation-based prompts. This strategy inherits English-centric framing and style and neglects cultural dimensions, ultimately constraining model generalization. We argue that the overlooked prompt space-the very inputs that define training distributions-offers a more powerful lever for improving multilingual performance. We introduce a lightweight framework for prompt-space optimization, where translated prompts are systematically transformed for Naturalness, Cultural Adaptation, and Difficulty Enhancement. Using an off-the-shelf multilingual LLM, we apply these transformations to prompts for 12 languages spanning 7 families. Under identical data conditions, our approaches achieve substantial and consistent downstream improvements over the translation-only baseline: +4.7% on Global-MMLU accuracy, +2.4% on Flores XCometXL and +35.3% wins in preferences on mArenaHard. We establish prompt-space optimization as a simple yet powerful paradigm for building multilingual LLMs that are more robust, culturally grounded, and globally capable.

## Summary

Here is a summary of the paper "The Art of Asking: Multilingual Prompt Optimization for Synthetic Data":

**Key Contribution:** This paper introduces a paradigm shift in multilingual synthetic data generation by optimizing the *prompt space* rather than just the completion space. The authors argue that current translation-based prompt expansion perpetuates English-centric framing and cultural biases, limiting model generalization. They propose a lightweight framework for systematically transforming translated prompts along three dimensions: Naturalness (removing translation artifacts), Cultural Adaptation (localizing content), and Difficulty Enhancement (increasing complexity).

**Methods:** The approach involves taking English user prompts, translating them into 12 target languages spanning diverse language families, and then applying transformation operators using Gemma3-27B as the transformation model. The transformations create optimized prompt distributions that are used to generate completions for fine-tuning a 7B parameter base model (CommandR7B). The method is evaluated across multiple benchmarks including translation (Flores), knowledge (Global-MMLU, Include44), mathematical reasoning (MGSM++), and open-ended generation (mArenaHard, PolyWrite).

**Results:** The prompt transformations consistently outperform translation-only baselines across all tasks and languages. The Cultural+Difficulty mixture achieved the most well-rounded performance with +4.7% on Global-MMLU, +2.4% on Flores XCometXL, and +35.3% win rates on mArenaHard. Key findings show that even minor prompt adjustments lead to substantial completion changes, cultural adaptation particularly helps knowledge tasks, and difficulty enhancement benefits mathematical reasoning and translation. The approach was especially effective for expanding language coverage to unsupported languages, with the Difficulty transformation bringing unsupported languages within 2 points of the much larger teacher model on translation tasks.

## Critique

Of course. Here is a critical assessment of the paper "The Art of Asking: Multilingual Prompt Optimization for Synthetic Data."

### **Summary of the Paper**

This paper proposes a paradigm shift in multilingual synthetic data generation. Instead of focusing solely on improving the quality of LLM-generated *completions* (the "generation space"), the authors argue for optimizing the *prompts* themselves (the "prompt space"). They introduce a framework to systematically transform machine-translated prompts along three dimensions: **Naturalness** (removing "translationese"), **Cultural Adaptation** (localizing content), and **Difficulty Enhancement** (making prompts more complex). They demonstrate that fine-tuning a 7B parameter model on data generated from these optimized prompts leads to significant and consistent improvements across a range of multilingual benchmarks compared to a baseline using only translated prompts.

---

### **Strengths**

1.  **Novelty and Conceptual Contribution:** The core idea—shifting the optimization focus from the generation space to the prompt space—is genuinely novel and impactful. While others have filtered or polished data post-generation, this paper is one of the first to systematically and proactively engineer the *input distribution* for multilingual data synthesis. This reframing is a valuable contribution to the field.

2.  **Significant and Consistent Results:** The empirical results are a major strength. The improvements are not marginal; they are substantial (+4.7% on G-MMLU, +35.3% win-rate on mArenaHard) and, crucially, are demonstrated to be **consistent** across 12 diverse languages. The fact that these gains are achieved by swapping out a relatively small number of prompts (10k per language) makes the result even more compelling, highlighting the outsized impact of prompt quality.

3.  **Rigorous and Multi-faceted Evaluation:** The evaluation is thorough and well-designed. The authors don't just rely on downstream metrics; they first analyze the data quality itself (perplexity, diversity, etc.) and then evaluate on a diverse suite of benchmarks covering discriminative, generative, close-ended, and open-ended tasks. This provides a holistic view of the method's effectiveness.

4.  **Practical Impact and Actionable Insights:** The paper provides clear, actionable insights:
    *   Going beyond simple translation is crucial; even perfect translation is limited by English-centric content.
    *   **Cultural Adaptation** is key for knowledge and culture-specific tasks.
    *   **Difficulty Enhancement** is highly effective for reasoning and translation, though it can reduce diversity.
    *   A **mix of Cultural and Difficulty** transformations yields the most well-rounded model.
    These findings provide a practical roadmap for others working on multilingual LLMs.

5.  **Analysis of Limitations:** The "Limitations" section is excellent and honest. It correctly identifies key issues like the inherent risks of synthetic data, potential evaluation biases (LLM judges may favor translationese), and the scope of their language selection. This strengthens the paper's credibility.

### **Weaknesses**

1.  **Clarity of Transformation Process:** While the three transformation types are well-motivated, the exact implementation is somewhat opaque. The prompts used for these transformations are relegated to an appendix. A more detailed discussion in the main text—perhaps with a few more concrete examples for each transformation type—would help the reader fully grasp the "how." The distinction between "Naturalness" and what a human would consider "post-editing" could be clearer.

2.  **Ablation and Cost Analysis:** The paper lacks a formal ablation study. For instance, it's noted that Cultural and Difficulty transformations are applied *on top of* the Naturalness-transformed prompts. What is the individual contribution of the Naturalness step to the final Cultural/Difficulty results? Furthermore, a discussion of the computational cost of this prompt-optimization pipeline versus a standard translation-based pipeline would be useful for practitioners considering its adoption.

3.  **The "Naturalness" Paradox:** One of the more puzzling results is that the "Naturalness" transformation, designed to reduce translationese, did not lead to a lower perplexity in the completions—a common proxy for naturalness. The authors offer a plausible hypothesis (bias in the perplexity scoring model), but this remains an unresolved point that slightly weakens the claim for this specific transformation. Its downstream gains were also the most modest.

4.  **Evaluation Benchmark Limitations (Acknowledged by Authors):** As the authors note, a key weakness lies in the evaluation benchmarks themselves. Many are translations of English prompts (e.g., mArenaHard++, PolyWrite), which may inherently favor models trained on more translated data. While they use a "naturalness-focused" judge for PolyWrite, the core task is still translated. Truly native, culturally-grounded evaluation sets are needed for ultimate validation, but the authors are transparent about this limitation.

### **Overall Assessment**

This is a **high-quality, impactful paper**. Its core conceptual contribution of "prompt-space optimization" is fresh and important. The methodology is sound, the empirical evidence is strong and extensive, and the practical implications are significant for anyone building multilingual AI systems. The weaknesses are relatively minor and mostly relate to a desire for more implementation detail and deeper ablation analysis. The paper convincingly demonstrates that "how you ask" is just as important as "what you get" when creating synthetic data for multilingual models. It is likely to inspire immediate follow-up work in the community.

---

# LoongRL:Reinforcement Learning for Advanced Reasoning over Long Contexts

Authors: Siyuan Wang, Gaokai Zhang, Li Lyna Zhang, Ning Shang, Fan Yang, Dongyao Chen, Mao Yang

Keywords: Long-context reasoning, Reinforcement learning, Multi-hop QA, KeyChain data, Plan-retrieve-reason-recheck pattern, Generalization to longer contexts

Comments: None

Paper link: [http://arxiv.org/abs/2510.19363v1](http://arxiv.org/abs/2510.19363v1)

## Abstract

Reasoning over long contexts is essential for large language models. While reinforcement learning (RL) enhances short-context reasoning by inducing "Aha" moments in chain-of-thought, the advanced thinking patterns required for long-context reasoning remain largely unexplored, and high-difficulty RL data are scarce. In this paper, we introduce LoongRL, a data-driven RL method for advanced long-context reasoning. Central to LoongRL is KeyChain, a synthesis approach that transforms short multi-hop QA into high-difficulty long-context tasks by inserting UUID chains that hide the true question among large collections of distracting documents. Solving these tasks requires the model to trace the correct chain step-by-step, identify the true question, retrieve relevant facts and reason over them to answer correctly. RL training on KeyChain data induces an emergent plan-retrieve-reason-recheck reasoning pattern that generalizes far beyond training length. Models trained at 16K effectively solve 128K tasks without prohibitive full-length RL rollout costs. On Qwen2.5-7B and 14B, LoongRL substantially improves long-context multi-hop QA accuracy by +23.5% and +21.1% absolute gains. The resulting LoongRL-14B reaches a score of 74.2, rivaling much larger frontier models such as o3-mini (74.5) and DeepSeek-R1 (74.9). It also improves long-context retrieval, passes all 128K needle-in-a-haystack stress tests, and preserves short-context reasoning capabilities.

## Summary

Of course. Here is a summary of the paper "LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts," focusing on its key contributions, methods, and results.

### Summary

This paper introduces **LoongRL**, a novel reinforcement learning (RL) framework designed to equip large language models (LLMs) with advanced reasoning capabilities for long-context tasks. The core challenge the authors address is that while modern LLMs can handle long contexts, they often excel at retrieval but struggle with deep, multi-step reasoning over that information. Existing RL methods for reasoning are primarily designed for short contexts and lack the high-quality, challenging data needed for long-context training.

### Key Contributions & Methods

1.  **KeyChain Data Synthesis:** The central innovation is **KeyChain**, a method for automatically generating high-difficulty, long-context reasoning data. It transforms standard multi-hop QA datasets (like HotpotQA) by:
    *   **Extending Context:** Padding the original context with many distracting documents to create long inputs (~16K tokens).
    *   **Inserting UUID Chains:** Hiding the original question within a chain of UUID key-value pairs embedded in the long context. The model must first trace this chain to discover the true question before it can even begin to reason and answer.
    This design forces the model to perform complex information seeking and reasoning, preventing it from relying on shallow retrieval or memorization.

2.  **Efficient RL Training:** The authors use the Group Relative Policy Optimization (GRPO) algorithm. A key design choice is training on 16K-token contexts, which is computationally feasible, and relying on the emergent reasoning patterns to **generalize to much longer contexts (up to 128K)** without costly full-length RL rollouts. They also use a simple but effective **two-way substring exact match** as a rule-based reward verifier to avoid reward hacking.

3.  **Emergent Reasoning Pattern:** RL training on KeyChain data induces a consistent, human-like **"plan–retrieve–reason–recheck"** reasoning pattern in the models. This structured approach leads to more reliable and logical solutions.

### Key Results

The method was evaluated on Qwen2.5-7B and 14B models, yielding state-of-the-art results:

*   **Superior Long-Context Reasoning:** LoongRL achieved massive improvements on long-context multi-hop QA benchmarks (LongBench), with **+23.5% and +21.1% absolute gains** for the 7B and 14B models, respectively.
*   **Competitive with Frontier Models:** The resulting **LoongRL-14B model scored 74.2**, rivaling much larger and more expensive frontier models like OpenAI's o3-mini (74.5) and DeepSeek-R1 (74.9).
*   **Strong Generalization:** Models trained only on 16K contexts demonstrated robust performance on tasks up to 128K tokens, showing effective length generalization.
*   **Preserved Capabilities:** Unlike some other reasoning-focused methods, LoongRL successfully preserved the model's original short-context reasoning (e.g., on MMLU) and even improved its long-context retrieval abilities, achieving a perfect pass on the "Needle-in-a-Haystack" benchmark.

In conclusion, LoongRL demonstrates that through carefully designed synthetic data (KeyChain) and efficient RL, smaller models can achieve frontier-level long-context reasoning by learning generalizable, structured thinking patterns.

## Critique

Of course. Here is a critique of the paper "LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts," covering its strengths, weaknesses, and overall impact.

### Summary

The paper "LoongRL" presents a novel reinforcement learning (RL) framework designed to enhance the long-context reasoning capabilities of large language models (LLMs). Its core innovation is **KeyChain**, a method for synthesizing high-difficulty training data from existing multi-hop QA datasets by embedding UUID chains that hide the true question within a long, distracting context. The key finding is that RL training on this data induces an emergent "plan–retrieve–reason–recheck" reasoning pattern in models, which generalizes effectively from a 16K-token training context to much longer contexts (up to 128K tokens).

### Strengths

1.  **High Novelty in Data Synthesis (KeyChain):** The KeyChain data construction is the paper's most significant contribution. It is a clever and well-motivated solution to the critical problem of scarce, high-quality RL data for long-context reasoning. By forcing the model to first trace a chain to uncover the question *before* reasoning, it directly incentivizes the complex, multi-step reasoning patterns the authors aim to elicit. This is a more sophisticated and targeted approach than simply padding contexts with irrelevant documents.

2.  **Impressive and Significant Results:** The performance gains are substantial. Achieving state-of-the-art results on standard benchmarks (LongBench) with models as small as 7B and 14B, and rivaling much larger frontier models like o3-mini and DeepSeek-R1, is a major accomplishment. The claim of **effective length generalization** (training on 16K, performing well on 128K) is particularly impactful, as it dramatically reduces the computational cost of long-context RL, which is typically prohibitive.

3.  **Strong Empirical Validation:** The paper is thoroughly evaluated across multiple dimensions: long-context reasoning, short-context reasoning preservation, and long-context retrieval. The inclusion of strong baselines (especially other RL-based and R1-distilled models) and extensive ablation studies (on KeyChain data and the reward verifier) makes the claims highly convincing.

4.  **Practical and Effective Reward Design:** The "two-way substring exact match" reward is a simple yet powerful solution to the challenge of reward hacking in open-ended QA. It strikes a good balance between the rigidity of exact match and the noise/variance of LLM-as-a-judge, and the ablation study convincingly demonstrates its superiority.

5.  **Clear Presentation of Emergent Behavior:** The paper does an excellent job of not just reporting metrics but also illustrating *how* the model's behavior changes. Figure 1 effectively contrasts the structured, human-like reasoning pattern of the LoongRL model with the entangled, error-prone pattern of the baseline, providing valuable insight into *why* the method works.

### Weaknesses

1.  **Limited Analysis of the Emergent Pattern:** While the "plan–retrieve–reason–recheck" pattern is highlighted, the paper provides limited investigation into its **robustness and consistency**. How often does this pattern emerge across different types of problems? Are there failure modes where the planning becomes overly verbose or circular? A more quantitative analysis of the reasoning traces (e.g., success rates correlated with the presence of a planning step) would strengthen this claim.

2.  **Narrow Focus on Multi-Hop QA:** The evaluation, while comprehensive within its scope, is heavily focused on multi-hop QA tasks. The generality of the KeyChain-induced reasoning pattern to other long-context tasks (e.g., **long-form summarization, code repository analysis, or legal document review**) remains an open question. Demonstrating effectiveness on a more diverse set of long-context tasks would significantly broaden the paper's impact.

3.  **Computational Cost Acknowledged but Not Quantified:** Although the 16K-training/128K-generalization paradigm is a major cost saver, the actual computational cost of the multi-stage RL process itself is not detailed. A rough estimate of GPU hours or a comparison of the total compute budget against other methods (like QwenLong-L1) would be helpful for practitioners to assess feasibility.

4.  **Potential Data Contamination Concern:** The seed datasets (HotpotQA, MuSiQue) are widely used, and some of the evaluation benchmarks (LongBench) are built from them. While the KeyChain transformation is substantial, the possibility of the base model having prior exposure to these corpora cannot be entirely ruled out. A brief discussion on this point would be prudent.

### Overall Assessment

**LoongRL is a highly compelling and impactful piece of research.** The KeyChain methodology is a novel and effective solution to a critical bottleneck in advancing long-context reasoning. The results are significant, demonstrating that with the right training data, smaller models can achieve performance competitive with frontier models at a fraction of the computational cost for training. The paper is well-written, with clear motivations, methodology, and strong empirical backing.

The main weakness lies in the yet-unproven generality of the approach beyond multi-hop QA-style tasks. However, the core insight—that synthesizing data to force specific cognitive behaviors (like planning and verification) can lead to robust, generalizable reasoning patterns—is a powerful contribution that will likely influence future work in both long-context processing and RL for LLMs.

