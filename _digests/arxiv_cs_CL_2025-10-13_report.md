---
title: "ArXiv Daily Digest on 2025-10-13"
collection: digests
permalink: /digests/arxiv_cs_CL_2025-10-13_report
date: 2025-10-13
location: "Online"
---

Today's literature highlights significant advances in multi-agent systems and model optimization, with several papers exploring how Large Language Models (LLMs) can collaborate effectively. Notable developments include **LLM×MapReduce-V3**, which introduces a hierarchically modular agent system using the **Model Context Protocol (MCP)** for dynamic, human-in-the-loop survey generation, and **StoryBox**, a hybrid bottom-up framework where agents interact in a simulated environment to produce coherent, long-form narratives. In optimization, **PerSyn (Personalized data Synthesis)** proposes a "Route then Generate" paradigm for multi-teacher distillation, efficiently assigning prompts to optimal teachers based on student learnability. Meanwhile, **Rollout Routing Replay (R3)** addresses instability in **Reinforcement Learning (RL)** for **Mixture-of-Experts (MoE)** models by aligning training and inference routers, preventing catastrophic collapse. Another study focuses on mitigating memorization risks during fine-tuning using n-gram-based early stopping and regularization. Together, these works underscore a trend toward more modular, efficient, and stable AI systems capable of complex, collaborative tasks.

## TL;DR

Total papers: 88 , Selected papers: 5

**TL;DR Summary of Recent arXiv Papers**

**Core Themes:**
- **Multi-Agent Systems & Collaboration**: Two papers explore multi-agent frameworks for complex tasks - one for academic survey generation (LLM×MapReduce-V3) and another for long-form story creation (StoryBox), demonstrating how specialized agents can collaborate effectively.
- **Training Efficiency & Stability**: Multiple papers address challenges in model training, including memorization mitigation during fine-tuning, personalized data synthesis for distillation, and stabilizing reinforcement learning in Mixture-of-Experts models.
- **Data Optimization**: Focus on improving how models learn from data, whether through better teacher selection, reducing memorization, or maintaining training-inference consistency.

**Key Insights:**

1. **Memorization Mitigation** (https://arxiv.org/abs/2510.11372): Shows memorization spikes early in fine-tuning and proposes n-gram based early stopping and regularization, reducing memorization by up to 40% with minimal performance loss.

2. **Modular Multi-Agent Systems** (https://arxiv.org/abs/2510.10890): Introduces MCP-based hierarchical agent system for survey generation, enabling dynamic workflow orchestration and human-in-the-loop interaction for better customization.

3. **Personalized Data Synthesis** (https://arxiv.org/abs/2510.10925): Proposes "Route then Generate" paradigm where prompts are routed to optimal teachers based on both quality and student learnability, outperforming traditional multi-teacher distillation.

4. **Emergent Story Generation** (https://arxiv.org/abs/2510.11618): Uses multi-agent simulation in sandbox environments to generate 10,000+ word stories through emergent character interactions, achieving state-of-the-art in coherence and creativity.

5. **MoE RL Stabilization** (https://arxiv.org/abs/2510.11370): Identifies routing discrepancies as source of RL instability in MoE models and solves with Rollout Routing Replay, preventing training collapse while maintaining efficiency.

**Trends**: Growing emphasis on modular, collaborative AI systems; increased focus on training efficiency and stability; novel approaches to data synthesis and knowledge transfer.

---

# Early Detection and Reduction of Memorisation for Domain Adaptation and Instruction Tuning

Authors: Dean L. Slack, Noura Al Moubayed

Keywords: memorisation mitigation, fine-tuning, domain adaptation, instruction tuning, n-gram memorisation, early stopping, loss regularisation, large language models

Comments: Accepted to Transactions of the ACL (TACL), 2025. 15 pages, 6
  figures, 3 tables

Paper link: [http://arxiv.org/abs/2510.11372v1](http://arxiv.org/abs/2510.11372v1)

## Abstract

Although large language models excel across many tasks, they can memorise training data and thereby expose private or copyrighted text. Most defences target the pre-training stage, leaving memorisation during fine-tuning, especially for domain adaptation and instruction tuning, poorly understood. We fine-tune Pythia, Llama3, and Mistral models spanning 1.4B-70B parameters on common evaluation datasets and track verbatim memorisation throughout training. We find that memorisation increases dramatically in the first few epochs, often significantly before either validation perplexity or evaluation performance is optimised. We use a simple but effective n-gram memorisation score which reliably precedes verbatim memorisation; using it as an early-stopping criterion mitigates memorisation with minimal performance loss. Further, we introduce an n-gram-aware loss regulariser and show that it reduces memorisation across all model families tested by up to 40% while minimising evaluation performance trade-offs when compared to an existing memorisation mitigation strategy. These results yield practical, scalable insights into memorisation dynamics during language model fine-tuning.

## Summary

This paper addresses the critical issue of memorization in large language models (LLMs) during fine-tuning for domain adaptation and instruction tuning. While most existing defenses target pre-training, this work focuses on understanding and mitigating memorization risks that emerge when adapting models to specialized or private datasets. The authors demonstrate that verbatim memorization of training data increases dramatically in early fine-tuning epochs, often before optimal validation perplexity or task performance is achieved.

The key contributions include: (1) analyzing memorization dynamics across different fine-tuning paradigms and model scales (1.4B-70B parameters), (2) introducing an n-gram memorization score as an effective precursor to verbatim memorization, (3) proposing optimal stopping criteria using n-gram thresholds to reduce memorization with minimal performance loss, and (4) developing an n-gram-aware loss regularizer that outperforms existing mitigation strategies.

Methodologically, the authors fine-tune models from Pythia, Llama, and Mistral families on diverse NLP datasets spanning classification, QA, summarization, and instruction following tasks. They employ k-extractable memorization metrics alongside their proposed n-gram memorization score, which measures partial phrase matching as an early warning signal.

Results show that the n-gram memorization score reliably predicts verbatim memorization across all model sizes and datasets, with particularly strong signals in domain adaptation and summarization tasks. Using n-gram thresholds for early stopping reduces memorization by approximately half compared to baseline approaches. More significantly, the proposed n-gram regularizer achieves up to 40% reduction in memorization while maintaining better performance trade-offs than the Goldfish regularization baseline. The approach scales effectively from smaller models to 70B-parameter architectures and generalizes across different task types and domains.

## Critique

Of course. Here is a critique of the paper "Early Detection and Reduction of Memorisation for Domain Adaptation and Instruction Tuning," focusing on its strengths and weaknesses.

### Strengths

1.  **High Practical Relevance and Significance:** The paper tackles a critical and timely issue in LLM deployment: data memorization leading to privacy and copyright risks. By focusing on the fine-tuning stage (domain adaptation and instruction tuning), it addresses a more common and resource-accessible scenario than pre-training, making the findings immediately applicable to many real-world use cases.

2.  **Systematic and Comprehensive Evaluation:**
    *   **Scale:** The experiments are extensive, covering multiple model families (Pythia, Llama, Mistral), a wide range of model sizes (1.4B to 70B parameters), and diverse datasets (classification, QA, summarization, instruction following).
    *   **Comparison:** The comparison between domain adaptation and instruction tuning provides valuable insights into how the framing of a task influences memorization.
    *   **Ablation:** The analysis of partial fine-tuning (freezing layers) adds depth to the understanding of memorization dynamics.

3.  **Novel and Pragmatic Contributions:**
    *   **n-gram Memorization as a Precursor:** The core idea of using a simple, computationally cheaper `n`-gram memorization score as an early warning signal for verbatim memorization is both novel and highly practical. It provides a deployable heuristic for practitioners.
    *   **Effective Mitigation Strategies:** The paper doesn't just diagnose the problem but offers solutions. The "Best n-gram" early stopping criterion is a simple yet effective mitigation strategy. The proposed `n`-gram regularizer is a more integrated solution that shows strong, scalable performance, often outperforming the existing Goldfish baseline.

4.  **Clear and Convincing Presentation:** The paper is well-structured, with a logical flow from problem identification to solution proposal and evaluation. The figures (especially Figures 2, 3, and 5) are effective at visually demonstrating the key findings, such as the predictive power of the `n`-gram score and the trade-offs between different mitigation strategies.

### Weaknesses

1.  **Limited Exploration of the Proposed Regularizer:** While the `n`-gram regularizer shows promising results, its description is relegated to the appendix. A more detailed discussion in the main text about its formulation, the intuition behind the confidence threshold, and its computational overhead compared to standard fine-tuning would strengthen the paper. The requirement to run the original pre-trained model for inference is a non-trivial cost that deserves more emphasis.

2.  **Narrow Scope of Decoding and Evaluation:**
    *   As noted in the limitations, the exclusive use of greedy decoding is a significant constraint. Memorization behavior under stochastic decoding (e.g., top-k, nucleus sampling) is highly relevant for real-world generative applications and remains an open question.
    *   The evaluation of "performance" is primarily task-specific accuracy. A broader analysis of how mitigation strategies affect other qualities like fluency, coherence, or creativity would provide a more holistic view of the trade-offs.

3.  **Superficial Categorical Analysis:** The analysis of memorization by semantic category (Figure 6) is interesting but somewhat surface-level. It identifies that medical and entity-related text is high-risk but offers little explanation or mechanistic insight into *why* these categories are more susceptible. A deeper dive into the linguistic or statistical properties of these categories (e.g., repetitiveness, specificity, term frequency) would be more insightful.

4.  **Claim of Generality vs. Dataset Constraints:** The paper argues for the generality of its findings but uses a maximum of only 5,000 fine-tuning samples. While this is justified for simulating small, private datasets, it limits the claim's scope. It's unclear if the same dynamics and mitigation efficacy hold when fine-tuning on much larger datasets (e.g., 50k+ samples), where overfitting might manifest differently.

### Overall Assessment

This is a strong, practical, and well-executed paper. Its primary strength lies in identifying a simple, scalable proxy (`n`-gram memorization) for a complex problem and leveraging it to develop effective mitigation strategies (early stopping, regularizer). The comprehensive empirical evaluation across models and tasks makes the findings robust and convincing. The main weaknesses are related to the scope of the experimental setup (decoding, dataset size) and a somewhat shallow treatment of the proposed regularizer and the categorical analysis. Despite these limitations, the paper makes a significant contribution by providing practitioners with actionable tools and insights to reduce memorization risks during fine-tuning.

---

# LLM$\times$MapReduce-V3: Enabling Interactive In-Depth Survey Generation through a MCP-Driven Hierarchically Modular Agent System

Authors: Yu Chao, Siyu Lin, xiaorong wang, Zhu Zhang, Zihan Zhou, Haoyu Wang, Shuo Wang, Jie Zhou, Zhiyuan Liu, Maosong Sun

Keywords: Multi-agent systems, Survey generation, Model Context Protocol, Modular agent architecture, Hierarchical planning, Human-in-the-loop interaction

Comments: Accepted by EMNLP2025 System Demonstration

Paper link: [http://arxiv.org/abs/2510.10890v1](http://arxiv.org/abs/2510.10890v1)

## Abstract

We introduce LLM x MapReduce-V3, a hierarchically modular agent system designed for long-form survey generation. Building on the prior work, LLM x MapReduce-V2, this version incorporates a multi-agent architecture where individual functional components, such as skeleton initialization, digest construction, and skeleton refinement, are implemented as independent model-context-protocol (MCP) servers. These atomic servers can be aggregated into higher-level servers, creating a hierarchically structured system. A high-level planner agent dynamically orchestrates the workflow by selecting appropriate modules based on their MCP tool descriptions and the execution history. This modular decomposition facilitates human-in-the-loop intervention, affording users greater control and customization over the research process. Through a multi-turn interaction, the system precisely captures the intended research perspectives to generate a comprehensive skeleton, which is then developed into an in-depth survey. Human evaluations demonstrate that our system surpasses representative baselines in both content depth and length, highlighting the strength of MCP-based modular planning.

## Summary

Based on the provided paper, here is a summary focusing on its key contributions, methods, and results.

**Key Contributions:**
This paper introduces LLM×MapReduce-V3, a hierarchically modular agent system designed for generating long-form academic surveys. Its primary contributions are threefold. Methodologically, it is the first to propose a Model Context Protocol (MCP)-based modular agent system for this task, enabling unprecedented customization and institutional integration. Architecturally, it advances the field with a dynamic, LLM-driven planner that orchestrates modules for adaptive, non-linear workflows, moving beyond rigid pipelines. From a design perspective, it is user-centric, incorporating a human-in-the-loop interaction framework to ensure the generated surveys align with the user's specific expertise and research perspectives.

**Methods:**
The system employs a multi-agent architecture where specialized agents (Analysis, Skeleton, and Writing Agents) handle distinct phases of survey generation. The core innovation is decomposing the algorithmic workflow into independent, composable MCP servers (e.g., Group Server, Skeleton Initialization Server, Digest Server). A central Orchestra Server, acting as a planner, dynamically selects and invokes these modules based on the execution history and available tool descriptions, enabling a flexible and adaptive process. The workflow involves literature retrieval and grouping, followed by an iterative process of skeleton initialization, content-aware digest construction, and multi-layer refinement inspired by convolutional networks. Crucially, the system integrates human feedback at multiple stages, including a topic consensus phase and an outline refinement phase, to precisely capture user intent.

**Results:**
The system was evaluated against other deep research systems (Gemini DeepResearch and Manus AI) through a human evaluation by domain experts across eleven topics. The results, judged on skeleton quality, content length, and overall quality, demonstrate the effectiveness of the proposed approach. LLM×MapReduce-V3 significantly outperformed the others in generating longer, more comprehensive content (winning 81.81% of votes for length) and achieved the highest quality (57.14% of votes). While Gemini DeepResearch was strong in producing coherent skeletons, the proposed system provided superior breadth of coverage and depth in literature review, highlighting the strength of its MCP-based modular planning and human-in-the-loop design.

## Critique

Of course. Here is a critique of the strengths and weaknesses of the paper "LLM×MapReduce-V3: Enabling Interactive In-Depth Survey Generation through a MCP-Driven Hierarchically Modular Agent System".

### Strengths

1.  **High Novelty in Architecture:** The paper's core contribution is highly novel. It is one of the first to systematically apply the Model Context Protocol (MCP) to build a complex, multi-agent system for a specific, challenging task like academic survey generation. The concept of decomposing a full pipeline into independent, composable MCP servers that can be dynamically orchestrated by a planner is a significant architectural advancement over fixed, monolithic pipelines.

2.  **Practical Significance and User-Centric Design:** The focus on modularity, replaceable agents (like the Search Agent), and support for user-defined extensions addresses a critical practical limitation of many AI research tools: inflexibility. This design makes the system highly adaptable for institutional use, domain-specific research, and integration with proprietary tools, greatly enhancing its potential real-world impact.

3.  **Comprehensive System Design:** The paper presents a well-thought-out and detailed system architecture. The clear separation of agents (Analysis, Skeleton, Writing) and servers (Group, Orchestra, Digest, etc.), along with the formalized notation for agent-server interactions, demonstrates a rigorous engineering approach. The "human-in-the-loop" mechanism for achieving consensus and integrating feedback is a crucial feature for ensuring the output aligns with user intent.

4.  **Strong Empirical Results:** The human evaluation, while limited in scale, shows compelling results. The system's decisive advantage in generating longer, more comprehensive surveys (81.81% win in "Length") and its lead in "Quality" (57.14%) against strong commercial baselines like Gemini DeepResearch provide concrete evidence of its effectiveness. This suggests the modular, iterative approach leads to more in-depth content synthesis.

### Weaknesses

1.  **Limited and Potentially Biased Evaluation:**
    *   **Scale:** The evaluation is based on only five reviewers and eleven topics, which is a relatively small sample size for making robust claims.
    *   **Baselines:** The choice of baselines is narrow. Comparing against only two commercial systems (Gemini DR and Manus AI) omits comparisons with other open-source academic survey systems mentioned in the related work (e.g., AutoSurvey, InteractiveSurvey). This makes it difficult to judge the specific improvement over the state-of-the-art in its own niche. The claim of being "first" would be stronger with direct comparisons.
    *   **Metric Clarity:** The evaluation criteria ("Skeleton," "Length," "Quality") are not rigorously defined. What constitutes "Quality" versus "Skeleton" is ambiguous, and the results are presented as a simple vote percentage without statistical significance testing.

2.  **Clarity of the "Orchestra Server":** While the concept of a dynamic planner is a strength, its implementation is one of the less clear parts of the paper. The description of the Orchestra Server is somewhat abstract. A concrete example of its decision-making process—showing the history, context, and the specific sequence of tools it chose to invoke—would make this core innovation much more tangible and easier to understand.

3.  **Under-specification of Technical Details:**
    *   **LLM Backbone:** The paper does not specify which specific LLMs were used to power the various agents and servers. This is a critical detail for reproducibility and for understanding the computational requirements and potential performance bottlenecks of the system.
    *   **Clustering Algorithm:** The Group Server's clustering algorithm is mentioned but not described. The choice of algorithm and its parameters can significantly impact the initial document grouping and thus the entire downstream process.

4.  **Presentation of the Core Workflow:** Figure 1 provides a high-level overview, but the flow of data and control between the various MCP servers during the critical Skeleton Refinement phase (Section 4.1.6) could be more clearly illustrated. A sequence diagram or a more detailed pipeline figure specifically for the Skeleton Agent's operation would enhance clarity.

### Summary

This paper presents a highly novel and architecturally sophisticated system that makes a compelling case for using MCP to create flexible, powerful, and user-centric AI research assistants. Its strengths lie in its innovative modular design, practical focus on customization, and promising initial results demonstrating its ability to produce comprehensive survey content.

The main weaknesses are related to the evaluation, which, while positive, is not comprehensive enough to fully validate the claims against the most relevant prior work. Furthermore, the paper would benefit from providing more specific technical details and clarifying the inner workings of its dynamic planner. Despite these shortcomings, the work represents a significant and valuable contribution to the field of AI-powered research and multi-agent systems.

---

# Find Your Optimal Teacher: Personalized Data Synthesis via Router-Guided Multi-Teacher Distillation

Authors: Hengyuan Zhang, Shiping Yang, Xiao Liang, Chenming Shang, Yuxuan Jiang, Chaofan Tao, Jing Xiong, Hayden Kwok-Hay So, Ruobing Xie, Angel X. Chang, Ngai Wong

Keywords: Personalized Data Synthesis, Multi-Teacher Distillation, Router-Guided Learning, Learnability-Quality Trade-off, Efficient Data Generation

Comments: 19 pages, 10 figures

Paper link: [http://arxiv.org/abs/2510.10925v1](http://arxiv.org/abs/2510.10925v1)

## Abstract

Training student models on synthetic data generated by strong teacher models is a promising way to distilling the capabilities of teachers. However, recent studies show that stronger models are not always optimal teachers, revealing a mismatch between teacher outputs and student learnability. To address this issue, we propose PerSyn (Personalized data Synthesis), a novel synthesis strategy that operates under a new ``Route then Generate'' paradigm to create data tailored to each student model, enabling it to learn more effectively. Specifically, PerSyn first assigns each prompt to its optimal teacher via a query-level router that jointly considers student learnability and teacher response quality. Each teacher then synthesizes data only for its assigned prompts, making the process more efficient than the conventional ``Generate then Select'' paradigm, where all teachers must generate parallel responses for the entire prompt set before constructing the final dataset. Extensive experiments across different model families and scales demonstrate that PerSyn consistently achieves superior or comparable performance to all baselines in instruct tuning and math reasoning settings. Further analysis verifies the effectiveness of PerSyn and offers extra insights to propel future research.

## Summary

Based on the provided paper, here is a summary focusing on its key contributions, methods, and results:

**Title:** Find Your Optimal Teacher: Personalized Data Synthesis via Router-Guided Multi-Teacher Distillation

**Key Problem:** The paper addresses a fundamental issue in knowledge distillation for Large Language Models (LLMs): stronger teacher models are not always optimal for training smaller student models. This is because the outputs of very powerful teachers can be too complex or distributionally mismatched for the student to learn from effectively (a "learnability gap"). Existing methods that mix data from strong and weak teachers or select a single teacher for the entire dataset are inefficient and fail to optimize learning at the individual prompt level.

**Proposed Method (PerSyn):** The authors introduce PerSyn, a novel data synthesis strategy based on a "Route then Generate" paradigm. Instead of having all teacher models generate responses for all prompts and then selecting the best ones ("Generate then Select"), PerSyn uses a router to assign each prompt to its single optimal teacher model before any generation occurs. The optimal teacher is selected based on a weighted combination of two rewards:
1.  **Quality Reward:** Assesses the inherent quality of a teacher's potential response (estimated by a reward model).
2.  **Learnability Reward:** Measures how easily the specific student model can learn from the teacher's response (calculated using the student's log-likelihood).
A PerSyn router is trained on a small subset of prompts with parallel teacher responses to learn this student-specific preference, enabling efficient, large-scale dataset construction.

**Key Results:**
*   **State-of-the-Art Performance:** Extensive experiments on instruction tuning (e.g., IFEval, TruthfulQA) and mathematical reasoning (e.g., MATH, GSM8K) tasks show that PerSyn consistently outperforms all baseline methods (Strong, Mix, Family-Strong, CAR) across various student model families (Qwen, Gemma, Llama) and scales (0.5B to 14B parameters).
*   **Efficiency:** The "Route then Generate" paradigm is significantly more efficient than "Generate then Select," as it avoids the cost of generating parallel responses from all teachers for the entire prompt set.
*   **Valuable Insights:** Analysis revealed that: 1) Both quality and learnability are crucial, with quality being more critical; 2) Surprisingly, smaller teacher models are often assigned more prompts than larger ones, indicating they are frequently more suitable teachers; 3) While Long-Chain-of-Thought (Long-CoT) models are rarely the optimal teacher, they remain necessary for handling a small subset of complex prompts.

**Conclusion:** PerSyn provides an effective and efficient framework for constructing personalized synthetic datasets, leading to more performant student models by intelligently matching each prompt to its optimal teacher based on a balance of response quality and student learnability.

## Critique

Of course. Here is a critique of the paper "Find Your Optimal Teacher: Personalized Data Synthesis via Router-Guided Multi-Teacher Distillation."

### Summary

This paper introduces **PerSyn**, a method for creating personalized synthetic datasets for knowledge distillation. It shifts the paradigm from "Generate then Select" (where all teachers generate responses for all prompts, and the best one is chosen) to "Route then Generate" (where a router first assigns each prompt to a single optimal teacher, which then generates the response). The optimal teacher is chosen based on a weighted combination of response quality and student learnability.

### Strengths

1.  **Novelty and Paradigm Shift:** The core idea of "Route then Generate" is genuinely novel and addresses a clear inefficiency in existing multi-teacher distillation methods. Moving from a compute-heavy, parallel generation process to a streamlined, routed one is a significant conceptual and practical contribution.
2.  **Strong Empirical Results:** The paper provides extensive experimental validation across multiple model families (Qwen, Gemma, Llama), scales (0.5B to 14B), and tasks (instruction tuning, math reasoning). The consistent and sometimes substantial performance gains over strong baselines (Strong, Mix, CAR) are compelling and convincingly demonstrate the method's effectiveness.
3.  **Comprehensive Analysis:** The paper goes beyond just reporting main results. The ablation studies (Finding 2), analysis of the alpha parameter, investigation into router performance, and—most importantly—the deep dive into *which teachers get allocated which prompts* (Findings 3 & 4) provide valuable insights. The finding that smaller models are often better teachers and the nuanced role of Long-CoT models are particularly noteworthy.
4.  **Resource for the Community:** The release of **PerSyn-Math**, a dataset with parallel responses from 15 teacher models, is a valuable contribution that will facilitate future research in this area.
5.  **Clarity of Presentation:** The paper is generally well-structured. The problem is clearly motivated, the method is explained step-by-step with the aid of a helpful figure (Fig. 1), and the results are presented in a clear, tabular format.

### Weaknesses

1.  **Computational Cost of Router Training:** While PerSyn is more efficient than "Generate then Select" during the main data synthesis phase, the paper somewhat downplays the cost of obtaining the router itself. Training a separate 1.5B parameter router for *each student model* and *each task* requires a non-trivial amount of computation and, crucially, the creation of a labeled pairwise preference dataset (2.5K prompts with parallel generations). The cost-benefit analysis versus the baselines could be more explicit.
2.  **Limited Scope of Generalization:** As acknowledged in the Limitations section, the experiments are confined to instruction tuning and math reasoning. It is uncertain how well PerSyn would perform in domains like code generation, creative writing, or complex, multi-step planning tasks where the notion of "learnability" might be different. The assumption that a student's token-level log-likelihood is a good proxy for learnability may not hold universally.
3.  **Dependence on Reward Model Quality:** The method's performance is contingent on the quality of the external reward model used to score response quality. While an experiment in the appendix reportedly addresses this, its details and results are not in the main text, leaving a potential vulnerability unexplored for the average reader. A weak or biased reward model could lead the router to make poor assignments.
4.  **Clarity on Router Specificity:** The requirement for a *separate* router per student per task is a key implementation detail that could be emphasized more in the main text. This specificity is both a strength (highly personalized) and a weakness (less flexible and more costly to scale).

### Overall Assessment

This is a **strong and impactful paper**. It identifies a clear problem in the current approach to data distillation, proposes a novel and intuitive solution, and backs it up with thorough experimentation and insightful analysis. The "Route then Generate" paradigm is a meaningful contribution that is likely to influence future work.

The weaknesses are primarily related to the practical costs of deployment (router training) and the yet-to-be-proven generality of the approach beyond the tested domains. However, these do not detract from the significance of the results presented. The paper is well-written, the methodology is sound, and the findings offer valuable new perspectives on the teacher-student relationship in LLM distillation.

---

# StoryBox: Collaborative Multi-Agent Simulation for Hybrid Bottom-Up Long-Form Story Generation Using Large Language Models

Authors: Zehao Chen, Rong Pan, Haoran Li

Keywords: Multi-Agent Simulation, Story Generation, Large Language Models, Long-Form Text Generation, Hybrid Bottom-Up Approach

Comments: Project: https://storyboxproject.github.io

Paper link: [http://arxiv.org/abs/2510.11618v1](http://arxiv.org/abs/2510.11618v1)

## Abstract

Human writers often begin their stories with an overarching mental scene, where they envision the interactions between characters and their environment. Inspired by this creative process, we propose a novel approach to long-form story generation, termed hybrid bottom-up long-form story generation, using multi-agent simulations. In our method, agents interact within a dynamic sandbox environment, where their behaviors and interactions with one another and the environment generate emergent events. These events form the foundation for the story, enabling organic character development and plot progression. Unlike traditional top-down approaches that impose rigid structures, our hybrid bottom-up approach allows for the natural unfolding of events, fostering more spontaneous and engaging storytelling. The system is capable of generating stories exceeding 10,000 words while maintaining coherence and consistency, addressing some of the key challenges faced by current story generation models. We achieve state-of-the-art performance across several metrics. This approach offers a scalable and innovative solution for creating dynamic, immersive long-form stories that evolve organically from agent-driven interactions.

## Summary

Based on the provided paper, here is a summary of "StoryBox: Collaborative Multi-Agent Simulation for Hybrid Bottom-Up Long-Form Story Generation Using Large Language Models":

**Key Contributions:**
The paper proposes a novel framework for long-form story generation that combines multi-agent simulation with a guided storytelling process. Its main contributions include: (1) a multi-agent simulation framework that generates dynamic story events through interactive agent behaviors, (2) a hybrid bottom-up approach that combines emergent interactions with guided storytelling for natural character development and plot progression, and (3) the ability to produce coherent stories exceeding 10,000 words, addressing key limitations of current story generation models.

**Methods:**
The system operates in two main phases. First, multiple LLM-based agents interact within a dynamic sandbox environment, where each character has defined core attributes (personality, daily plans) and an "Abnormal Behavior" attribute that introduces unpredictability. Agents generate events through movements, conversations, and interactions with the environment. Second, a Storyteller Agent processes these emergent events using a hybrid bottom-up workflow that includes event summarization, story information generation (type, title, themes), and iterative story generation. The approach uses dynamic windowing for event management and information retrieval to maintain coherence across long narratives.

**Results:**
Experimental evaluations against various baselines (vanilla LLMs, structured frameworks, and other multi-agent systems) show that StoryBox achieves state-of-the-art performance across multiple metrics. Both automatic and human evaluations (with 78 participants) consistently ranked StoryBox highest across dimensions including Plot, Creativity, Character Development, Language Use, Conflict Quality, and Overall quality. The system generates stories averaging around 12,000 words while maintaining coherence and consistency. Ablation studies confirmed the importance of key components like object descriptions, abnormal behaviors, and dynamic context windows. A simulation duration study found that 7 days of simulation strikes the optimal balance between quality and efficiency.

## Critique

Here's a critical assessment of the "StoryBox" paper:

**Strengths:**

1. **Novelty of Approach**: The hybrid bottom-up methodology combining multi-agent simulation with story generation is genuinely innovative. The concept of using emergent agent interactions as a foundation for storytelling, rather than imposing rigid top-down structures, represents a fresh perspective in narrative generation.

2. **Comprehensive Evaluation**: The paper employs both automatic and human evaluation across multiple dimensions (Plot, Creativity, Character Development, etc.), providing robust evidence for the system's performance. The inclusion of 78 human evaluators from diverse backgrounds adds credibility to the results.

3. **Technical Depth**: The system demonstrates sophisticated technical implementation, including dynamic windowing mechanisms, event summarization strategies, and character behavior modeling with abnormal behavior attributes. The ability to generate 10,000+ word stories while maintaining coherence is impressive.

4. **Ablation Studies**: The systematic ablation study provides valuable insights into the contribution of individual components (object descriptions, abnormal behaviors, dynamic context window), strengthening the methodological claims.

**Weaknesses:**

1. **Limited Baseline Comparison**: While the paper compares against several methods, the selection of baselines could be more comprehensive. Some recent state-of-the-art story generation systems may be missing, and the comparison with "vanilla LLMs" might not fully represent the current competitive landscape.

2. **Evaluation Subjectivity**: Despite the multi-dimensional evaluation framework, story quality assessment remains inherently subjective. The paper acknowledges this limitation but doesn't sufficiently address how to mitigate the inherent biases in both human and automatic evaluation.

3. **Scalability Concerns**: The sequential nature of the simulation and the computational costs (token usage doubling with duration increases) raise practical concerns about scalability. The paper mentions parallelization challenges but doesn't offer concrete solutions.

4. **Dataset Limitations**: The authors created a new dataset but provide limited details about its construction methodology, size, and diversity. More transparency about dataset characteristics would strengthen the experimental validity.

**Presentation Clarity:**

The paper is generally well-structured with clear figures that effectively illustrate the system architecture and results. However, some technical details are relegated to supplementary material, which may hinder full reproducibility. The writing is professional but occasionally dense in methodological sections.

**Significance of Results:**

The state-of-the-art performance across multiple metrics, particularly in human evaluation, demonstrates the practical value of the approach. The ability to generate long-form coherent stories addresses a significant challenge in narrative generation. The simulation duration study provides useful practical guidance for implementation.

Overall, this represents a substantial contribution to the field of automated story generation, with a novel methodology and compelling empirical results, though some implementation and evaluation aspects could be further strengthened.

---

# Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers

Authors: Wenhan Ma, Hailin Zhang, Liang Zhao, Yifan Song, Yudong Wang, Zhifang Sui, Fuli Luo

Keywords: Mixture-of-Experts, Reinforcement Learning, Training-Inference Discrepancies, Routing Distribution, Rollout Routing Replay

Comments: None

Paper link: [http://arxiv.org/abs/2510.11370v1](http://arxiv.org/abs/2510.11370v1)

## Abstract

Reinforcement learning (RL) has emerged as a crucial approach for enhancing the capabilities of large language models. However, in Mixture-of-Experts (MoE) models, the routing mechanism often introduces instability, even leading to catastrophic RL training collapse. We analyze the training-inference consistency of MoE models and identify a notable discrepancy in routing behaviors between the two phases. Moreover, even under identical conditions, the routing framework can yield divergent expert selections across repeated forward passes. To address this foundational inconsistency, we propose Rollout Routing Replay (R3), a method that records routing distributions from the inference engine and replays them during training. R3 significantly reduces training-inference policy KL divergence and mitigates extreme discrepancies without compromising training speed. Extensive experiments on various settings confirm that R3 succeeds in stabilizing RL training, preventing collapse and outperforming methods such as GSPO and TIS. We believe this work can offer a new solution for stabilizing RL in MoE models.

## Summary

This paper addresses the critical challenge of training instability in Mixture-of-Experts (MoE) models during reinforcement learning (RL). The authors identify that the primary source of instability stems from routing discrepancies between training and inference phases—where MoE routers select different experts for the same tokens across different frameworks, leading to significant policy mismatches and even catastrophic training collapse.

The key contribution is **Rollout Routing Replay (R3)**, a simple yet effective method that records routing distributions from the inference engine and replays them during training. This alignment ensures consistent expert selection while preserving gradient flow by applying softmax to training logits over the replayed expert masks. R3 also supports efficient integration with KV cache mechanisms for multi-turn dialogues, making it practical for real-world applications.

Experimental results on mathematical reasoning tasks using Qwen3-30B-A3B models demonstrate that R3 significantly reduces training-inference KL divergence (from 1.5×10⁻³ to 7.5×10⁻⁴) and extreme token discrepancies by an order of magnitude. Compared to baseline methods like GSPO and TIS, R3 consistently improves training stability, prevents collapse, and achieves superior performance across various RL settings, including both single and multiple mini-step configurations. The method proves particularly effective in stabilizing MoE RL training without compromising efficiency.

## Critique

Of course. Here is a critique of the paper "Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers," focusing on its strengths and weaknesses.

### Strengths

1.  **Novelty and Problem Identification:** The paper's primary strength is its clear identification of a specific and critical problem: the instability in Reinforcement Learning (RL) for Mixture-of-Experts (MoE) models stems from **routing discrepancies** between the inference and training engines. While training-inference inconsistency is a known issue in LLM RL, this work is the first to systematically analyze and attribute it to the non-continuous, discrete routing mechanism in MoEs. The novelty lies in targeting the root cause (the router) rather than applying a patch to the symptoms (the policy outputs).

2.  **Elegant and Practical Solution:** The proposed method, **Rollout Routing Replay (R3)**, is simple, elegant, and computationally lightweight. By caching the binary expert selection masks from inference and replaying them during the training forward pass, it directly enforces alignment. Crucially, it preserves gradient flow by recalculating the gating weights using the *training* logits, allowing the router to still be optimized. The integration with existing KV-cache mechanisms for multi-turn dialogues is a thoughtful and practical design choice.

3.  **Rigorous and Comprehensive Evaluation:** The paper provides strong empirical evidence to support its claims.
    *   **Diagnostic Analysis:** It begins with a thorough diagnostic (Section 3), quantifying the problem using KL divergence, scatter plots, and a newly introduced "Extreme Token Distribution" metric, convincingly showing that MoEs suffer from significantly higher discrepancies than dense models.
    *   **Ablation of R3:** It demonstrates that R3 effectively reduces the training-inference KL divergence of the MoE model to a level comparable to a dense baseline.
    *   **Performance and Stability:** The main experiments (Section 5) are extensive, testing R3 across different models (Base/SFT), different RL optimizers (GRPO, GSPO, TIS), and different training regimes (single/multi-mini-step). The results consistently show that R3 prevents training collapse and achieves superior or competitive final performance.

4.  **Clarity of Presentation:** The paper is generally well-written and easy to follow. The problem is motivated clearly, the method is explained with precise notation, and the figures (especially Figures 2 and 5) effectively illustrate the core problem and the solution's impact.

### Weaknesses

1.  **Limited Exploration of Root Cause for Router Divergence:** While the paper excellently identifies *that* the routers diverge, it provides less insight into *why* they diverge. The explanation ("small perturbations in the router input can lead to entirely different experts being selected") is plausible but not deeply investigated. Is this due to numerical precision differences between SGLang and Megatron? Non-deterministic operations? The sensitivity of the top-k operation? A deeper analysis here would strengthen the foundational contribution.

2.  **Potential Long-Term Impact on Router Learning:** A theoretical weakness is the potential long-term effect of R3 on the router's learning dynamics. By forcing the training-phase router to use the inference-phase mask, are we potentially stifling its ability to explore and discover better expert selections during training? The paper shows the router still receives gradients, but it's unclear if this "teacher forcing" of the mask could lead to a sub-optimal routing policy in the very long run. The experiments, while comprehensive, are not long enough to fully rule this out.

3.  **Overstated Synergy with Other Methods:** The results show that R3 alone is highly effective, but its combination with other methods like TIS does not yield clear gains and can sometimes be detrimental. The paper correctly notes this is likely because R3 already solves the core problem TIS addresses. However, the initial framing might suggest broader compatibility, while the results indicate that R3 may simply **supersede** the need for these other methods in the context of MoE models, rather than being complementary.

4.  **Scope of Models and Tasks:** The evaluation is based solely on the Qwen2-MoE model family and mathematical reasoning tasks. While this is a valid and challenging domain, it would be more compelling to see results on a wider variety of models (e.g., a SwitchTransformer or Mixtral-based architecture) and tasks (e.g., code generation or dialogue) to demonstrate the generalizability of the approach.

### Overall Assessment

This is a **high-quality, impactful paper**. It identifies a precise and important problem that has likely hindered progress in scaling RL to large MoE models. The proposed R3 method is a simple, low-overhead, and highly effective solution that directly addresses the root cause. The empirical evaluation is thorough and convincing, demonstrating significant improvements in training stability and performance.

The main weaknesses are relatively minor and relate to the depth of the root-cause analysis and the exploration of long-term effects. The core contribution—identifying router misalignment as the key instability factor in MoE-RL and providing a practical fix—is significant and likely to be adopted by practitioners in the field. The clarity of the presentation makes the work accessible and its benefits easy to understand.

