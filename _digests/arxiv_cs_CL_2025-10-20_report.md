---
title: "ArXiv Daily Digest on 2025-10-20"
collection: digests
permalink: /digests/arxiv_cs_CL_2025-10-20_report
date: 2025-10-20
location: "Online"
---

Today's research landscape showcases significant advancements in multi-agent systems and efficient model architectures, with several papers introducing novel frameworks for enhanced collaboration and scalability. A prominent theme is verification-aware planning, exemplified by VeriMAP, which integrates explicit verification functions into multi-agent workflows to improve robustness in complex reasoning tasks. Another key development is Enterprise Deep Research (EDR), a steerable multi-agent system that enables human-in-the-loop guidance for enterprise analytics, demonstrating the growing emphasis on controllable AI systems. In architectural innovations, ReXMoE (Reusing Experts with Mixture-of-Experts) presents a novel approach to Mixture-of-Experts (MoE) by enabling cross-layer expert reuse, while Contextual Attention Modulation (CAM) offers a new parameter-efficient fine-tuning method for multi-task adaptation in Large Language Models (LLMs). Complementing these, QueST introduces a sophisticated framework for generating challenging synthetic coding problems through difficulty-aware training, addressing the critical bottleneck of high-quality training data for reasoning tasks.

## TL;DR

Total papers: 61 , Selected papers: 5

Here's a TL;DR summary of the recent arXiv papers on efficient LLM architectures and multi-agent systems:

**Main Themes:**
- **Parameter-Efficient Scaling**: Multiple papers explore novel Mixture-of-Experts (MoE) architectures for scaling LLMs efficiently. ReXMoE introduces cross-layer expert reuse to expand routing flexibility without parameter inflation (https://arxiv.org/abs/2510.17483v1), while HyCAM uses contextual attention modulation for multi-task adaptation (https://arxiv.org/abs/2510.17705v1).

- **Multi-Agent Systems & Verification**: VeriMAP addresses coordination challenges in multi-agent systems through verification-aware planning with automatic validation functions (https://arxiv.org/abs/2510.17109v1). Enterprise Deep Research demonstrates steerable multi-agent systems for enterprise analytics with human-in-the-loop guidance (https://arxiv.org/abs/2510.17797v1).

- **Synthetic Data Generation**: QueST presents a novel framework for generating challenging coding problems through difficulty-aware sampling and rejection fine-tuning, enabling significant performance gains in code reasoning (https://arxiv.org/abs/2510.17715v1).

**Key Insights:**
These works collectively push toward more efficient, verifiable, and scalable LLM systems through architectural innovations (MoE variants), improved coordination mechanisms (verification-aware planning), and automated data generation pipelines - all while maintaining or reducing computational costs compared to conventional approaches.

---

# Contextual Attention Modulation: Towards Efficient Multi-Task Adaptation in Large Language Models

Authors: Dayan Pan, Zhaoyang Fu, Jingyuan Wang, Xiao Han, Yue Zhu, Xiangyu Zhao

Keywords: Large Language Models, Parameter-Efficient Fine-Tuning, Multi-Task Adaptation, Contextual Attention Modulation, Self-Attention Mechanisms

Comments: Accepted by CIKM' 25

Paper link: [http://arxiv.org/abs/2510.17705v1](http://arxiv.org/abs/2510.17705v1)

## Abstract

Large Language Models (LLMs) possess remarkable generalization capabilities but struggle with multi-task adaptation, particularly in balancing knowledge retention with task-specific specialization. Conventional fine-tuning methods suffer from catastrophic forgetting and substantial resource consumption, while existing parameter-efficient methods perform suboptimally in complex multi-task scenarios. To address this, we propose Contextual Attention Modulation (CAM), a novel mechanism that dynamically modulates the representations of self-attention modules in LLMs. CAM enhances task-specific features while preserving general knowledge, thereby facilitating more effective and efficient adaptation. For effective multi-task adaptation, CAM is integrated into our Hybrid Contextual Attention Modulation (HyCAM) framework, which combines a shared, full-parameter CAM module with multiple specialized, lightweight CAM modules, enhanced by a dynamic routing strategy for adaptive knowledge fusion. Extensive experiments on heterogeneous tasks, including question answering, code generation, and logical reasoning, demonstrate that our approach significantly outperforms existing approaches, achieving an average performance improvement of 3.65%. The implemented code and data are available to ease reproducibility at https://github.com/Applied-Machine-Learning-Lab/HyCAM.

## Summary

Based on the provided paper, here is a summary of its key contributions, methods, and results:

**Key Contributions:**
The paper introduces a novel framework for efficient multi-task adaptation in Large Language Models (LLMs). The primary contributions are the **Contextual Attention Modulation (CAM)** mechanism and the **Hybrid Contextual Attention Modulation (HyCAM)** framework. CAM is designed to dynamically modulate the representations within a model's self-attention modules based on the input context, aiming to enhance task-specific features while preserving the model's pre-trained general knowledge. The HyCAM framework operationalizes CAM by combining a shared, full-parameter CAM module (for cross-task knowledge) with multiple specialized, lightweight CAM modules (for task-specific features), using a dynamic routing strategy for adaptive knowledge fusion.

**Methods:**
The core **CAM mechanism** works by generating a context-dependent modulation weight tensor from the normalized hidden state of a Transformer layer. This tensor is applied via an element-wise product to the output of the self-attention module, selectively amplifying or suppressing attentional signals to refine the contextual information flow for a given task. The **HyCAM framework** builds on this by integrating a hybrid architecture: a single shared CAM module captures common patterns, while multiple specialized CAM modules (implemented using a parameter-efficient technique, SLoRA) learn distinct features for different tasks. A Gumbel-Softmax-based router dynamically weights the contributions of the specialized modules for each input token, and a load-balancing loss encourages balanced utilization of these modules. The entire system is trained end-to-end with a composite loss function.

**Results:**
The authors conducted extensive experiments on a multi-task benchmark comprising question answering, code generation, logical reasoning, and other domains, using various backbone LLMs like Llama, Mistral, and Qwen. The results demonstrate that HyCAM significantly outperforms existing state-of-the-art methods, including Full Fine-Tuning, LoRA, Multi LoRA, and RieMoE-LoRA. It achieved an average performance improvement of **3.65%** across all metrics and models, with lower perplexity and higher BLEU and ROUGE scores. The framework also showed consistent advantages across different model sizes and tasks, faster convergence, and improved representational coherence, as validated through ablation studies and qualitative analyses.

## Critique

Of course. Here is a critique of the paper "Contextual Attention Modulation: Towards Efficient Multi-Task Adaptation in Large Language Models," focusing on its strengths and weaknesses.

### **Overall Assessment**

This paper presents a well-structured and technically sound contribution to the field of efficient fine-tuning for Large Language Models (LLMs). The proposed HyCAM framework demonstrates strong empirical performance and is backed by a clear, motivated design. The work is timely, addressing the critical challenge of multi-task adaptation without catastrophic forgetting.

---

### **Strengths**

1.  **Novelty and Clear Motivation:** The core idea—**Contextual Attention Modulation (CAM)**—is novel and well-motivated. The authors provide a compelling rationale based on the established understanding of Transformer components: that Feed-Forward Networks (FFNs) are knowledge stores, while self-attention is responsible for contextual integration. Their insight to modulate the *attention outputs* to better integrate pre-trained knowledge with task-specific context, rather than overwriting parameters, is elegant and distinct from existing PEFT methods like LoRA which modify the weight matrices themselves.

2.  **Comprehensive and Well-Designed Framework:** The **HyCAM** framework is a sophisticated extension of the core CAM idea. The hybrid design, combining a single full-parameter Shared CAM (for common knowledge) with multiple parameter-efficient Specialized CAMs (for task-specific features), is a robust solution to the multi-task learning problem. The inclusion of a dynamic routing mechanism with a load-balancing loss is a necessary and well-executed component to manage the mixture-of-experts-style architecture effectively.

3.  **Extensive and Convincing Experimental Setup:** The evaluation is a major strength of the paper.
    *   **Diverse Benchmarks:** Using five datasets from distinct domains (reasoning, medical QA, instruction-following, code generation, IR-QA) provides a thorough test for multi-task adaptation.
    *   **Multiple Model Families:** Evaluating across Llama, Mistral, and Qwen families, and across different model sizes, strongly supports the generalizability of the approach.
    *   **Strong Baselines:** The choice of baselines (Full Fine-Tune, LoRA, Multi-LoRA, RieMoE-LoRA) is appropriate and covers the relevant state-of-the-art.
    *   **Statistically Significant Results:** The reported performance gains (average 3.65% improvement) are backed by statistical significance testing, adding credibility.

4.  **Strong Ablation Studies:** The ablation study (Table 6) is excellent. It systematically deconstructs the HyCAM framework, clearly demonstrating the contribution of each component (Shared CAM, Specialized CAMs, PEFT structure) to the final performance.

5.  **Clear Presentation:** The paper is generally well-written. The structure is logical, the methodology is described in detail with clear equations, and the figures effectively illustrate the architecture and qualitative results.

---

### **Weaknesses**

1.  **Limited Analysis of "Catastrophic Forgetting":** While the paper's motivation heavily emphasizes mitigating catastrophic forgetting and preserving general knowledge, the experiments primarily measure performance *on the fine-tuning tasks*. A more convincing demonstration would involve evaluating the models on a held-out set of *original, pre-training-like tasks* (e.g., standard benchmarks like MMLU or HellaSwag) to quantitatively show that HyCAM indeed forgets less than full fine-tuning or other baselines. The current evidence is largely implicit.

2.  **Scalability of the "Shared CAM" Module:** A potential weakness lies in the Shared CAM module, which uses a full-parameter (\(d \times d\)) projection matrix. While the overall framework is parameter-efficient compared to full fine-tuning, this component's parameter count scales quadratically with the hidden dimension \(d\). For extremely large models (e.g., 70B+ parameters), this could become a non-trivial cost. The paper would be strengthened by a discussion of this scaling and potential alternatives (e.g., a low-rank shared module for very large models).

3.  **Interpretation of Qualitative Results:** The qualitative evaluation (RQ5) is good but could be more insightful.
    *   Figure 2 (t-SNE plots) shows "more coherent clusters" but the connection to *improved task performance* is somewhat vague. A more direct analysis, perhaps showing how attention patterns change for task-specific keywords, would be stronger.
    *   Figure 4 (weight matrix) is difficult to interpret for a reader without knowing what the specific features or tokens are.

4.  **Clarity on Task Identity:** The paper operates in a multi-task setting where the model is trained on a mixture of tasks. However, it is not explicitly stated if the model receives an explicit task identifier during training or inference. The dynamic router is context-aware, but its ability to disentangle tasks without any prompt or identifier is a remarkable claim that could use more emphasis or analysis.

### **Summary**

This is a high-quality paper that introduces a novel and effective method for multi-task adaptation of LLMs. The strengths significantly outweigh the weaknesses. The proposed HyCAM framework is novel, well-designed, and demonstrates state-of-the-art performance through extensive and rigorous experiments. The main areas for improvement are providing more direct evidence for catastrophic forgetting mitigation and a deeper discussion of the scalability and computational footprint of the full-parameter Shared CAM component. This work represents a meaningful advance in the field of parameter-efficient fine-tuning.

---

# Verification-Aware Planning for Multi-Agent Systems

Authors: Tianyang Xu, Dan Zhang, Kushan Mitra, Estevam Hruschka

Keywords: multi-agent systems, verification-aware planning, LLM agents, task decomposition, verification functions, coordination, robustness

Comments: Submission for ARR Oct

Paper link: [http://arxiv.org/abs/2510.17109v1](http://arxiv.org/abs/2510.17109v1)

## Abstract

Large language model (LLM) agents are increasingly deployed to tackle complex tasks, often necessitating collaboration among multiple specialized agents. However, multi-agent collaboration introduces new challenges in planning, coordination, and verification. Execution failures frequently arise not from flawed reasoning alone, but from subtle misalignments in task interpretation, output format, or inter-agent handoffs. To address these challenges, we present VeriMAP, a framework for multi-agent collaboration with verification-aware planning. The VeriMAP planner decomposes tasks, models subtask dependencies, and encodes planner-defined passing criteria as subtask verification functions (VFs) in Python and natural language. We evaluate VeriMAP on diverse datasets, demonstrating that it outperforms both single- and multi-agent baselines while enhancing system robustness and interpretability. Our analysis highlights how verification-aware planning enables reliable coordination and iterative refinement in multi-agent systems, without relying on external labels or annotations.

## Summary

Of course. Here is a summary of the paper "Verification-Aware Planning for Multi-Agent Systems."

### Summary

This paper introduces **VeriMAP**, a novel framework designed to enhance the robustness and performance of multi-agent systems built on large language models (LLMs). The central problem it addresses is that in multi-agent collaboration, failures often occur not from incorrect reasoning alone, but from subtle misalignments in task interpretation, output formats, or inter-agent handoffs. VeriMAP tackles this by tightly integrating planning with contextualized verification.

The core contribution is a **verification-aware planner** that does more than just decompose a complex task into a directed acyclic graph (DAG) of subtasks. For each subtask, it also generates explicit **Verification Functions (VFs)**, which serve as precise "acceptance criteria." These VFs come in two forms:
1.  **Python VFs:** Programmatic assertions for checking structural and functional correctness (e.g., output type, code passing tests).
2.  **Natural Language VFs:** Instructions for an LLM-based verifier to assess semantic correctness for open-ended tasks.

The system operates through four modules:
1.  The **Planner** creates the task plan and its associated VFs.
2.  The **Executor** (using a smaller, cost-effective model) performs the subtask.
3.  The **Verifier** checks the executor's output against the planner-generated VFs.
4.  The **Coordinator** orchestrates the workflow, managing context, triggering retries upon VF failure, and initiating **replanning** if necessary.

### Key Results

The authors evaluated VeriMAP on five diverse datasets (MultiHopRAG, HumanEval, BigCodeBench-Hard, GSM8K, and Olympiads) spanning question answering, programming, and math. The results demonstrate that VeriMAP:

*   **Consistently outperforms both single-agent and multi-agent baselines.** It achieves state-of-the-art results, with particularly significant gains on challenging benchmarks (e.g., +9.8% on Olympiads and +4.05% on BigCodeBench-Hard compared to the next-best method).
*   **Effectively leverages replanning.** An ablation study (VeriMAP-1it) showed that the replanning mechanism is crucial for success. More importantly, VeriMAP's planner-generated VFs provide more informative failure signals, leading to much more effective replanning compared to a baseline with generic verification.
*   **Generates adaptive and effective verification.** Analysis showed that the planner intelligently tailors the type and complexity of VFs to the task domain (e.g., mostly Python VFs for coding, mostly NL VFs for QA). Furthermore, VeriMAP's VFs generally achieved a better balance between false positives and false negatives than a baseline verifier.
*   **Is cost-effective.** While incurring a higher cost than single-agent baselines, the cost increase is modest. The framework offsets this by delegating most execution to smaller models, making the performance gains well worth the marginal cost, especially for complex tasks.

In conclusion, VeriMAP presents a powerful framework that moves beyond simple task decomposition by embedding verification directly into the planning process. This approach significantly improves the reliability, interpretability, and final performance of multi-agent LLM systems on complex reasoning tasks.

## Critique

Of course. Here is a critique of the paper "Verification-Aware Planning for Multi-Agent Systems" (VeriMAP), commenting on its strengths and weaknesses.

### Overall Assessment

This is a strong, well-executed paper that addresses a clear and important problem in multi-agent LLM systems. The proposed framework, VeriMAP, is novel, the experimental evaluation is thorough, and the results are significant. The presentation is generally clear, with a logical structure and helpful visualizations.

---

### Strengths

**1. Novelty of the Approach:**
The core idea of **"verification-aware planning"** is the paper's most significant contribution. Instead of treating verification as a separate, post-hoc step or using generic checks, the planner actively generates specific, contextual Verification Functions (VFs) for each subtask *during the planning phase*. This tight integration is a clear advance over existing multi-agent systems (MAP, MAP-V) and single-agent methods (ReAct). The distinction between Python-based VFs (for structural/functional checks) and Natural Language VFs (for semantic checks) is a practical and well-motivated design choice that leverages the strengths of different verification modalities.

**2. Comprehensive and Convincing Evaluation:**
The experimental design is a major strength.
*   **Diverse Benchmarks:** The use of five datasets spanning RAG, Programming, and Math tasks demonstrates the generality of the approach beyond a single domain.
*   **Strong Baselines:** The paper includes relevant and strong baselines, including single-agent ReAct with powerful models (gpt-4.1, o3) and multi-agent systems (MAP, MAP-V). This allows for a fair and meaningful comparison.
*   **Ablation Studies:** The "-1it" variants (without replanning) effectively isolate and demonstrate the critical importance of the replanning loop, showing that VFs provide high-quality signals for this process.
*   **Multi-Model Analysis:** Testing with different planner LLMs (o3, Claude-Sonnet) adds robustness to the claims and provides interesting insights into model specialization.
*   **In-Depth Analysis:** The paper goes beyond simple accuracy metrics to include cost analysis, VF statistics, and a detailed error analysis of false positives/negatives, which provides a nuanced understanding of the system's behavior.

**3. Significance of Results:**
The results are not just incremental; they are substantial, particularly on the most challenging tasks. Improvements of **+9.8% on Olympiads** and **+4.05% on BigCodeBench-Hard** over a very strong single-agent baseline (ReAct gpt-4.1) are highly significant. The case study in Section 3.4 is particularly compelling, as it concretely illustrates how VeriMAP's structured VFs catch subtle errors that a generic verifier misses, directly leading to a correct final answer.

**4. Clarity of Presentation:**
The paper is well-structured and easy to follow.
*   Figure 1 provides an excellent high-level overview of the VeriMAP architecture.
*   The description of the four components (Planner, Executor, Verifier, Coordinator) is clear and logically sequenced.
*   The use of tables and figures is effective in presenting complex results.

---

### Weaknesses

**1. Clarity on the Planner's "Awareness":**
A minor point of confusion is the exact mechanism that makes the planner "verification-aware." The paper states the planner *generates* VFs, but it's not entirely clear *how* it knows what specific VFs to create. Is it purely based on the initial task instruction and the planner's own reasoning about potential failure modes? A slightly more detailed explanation of the prompting or reasoning process for the planner itself would strengthen this core concept.

**2. Cost-Benefit Justification:**
While the cost analysis is present and valuable, the discussion could be more nuanced. The paper rightly notes that VeriMAP's cost is higher than single-agent baselines but justified by the performance gains. However, for a practitioner, the cost increase (e.g., nearly 10x on GSM8K for a ~2% gain) might be hard to justify, whereas on Olympiads, the cost is much more comparable for a large gain. The paper could more strongly frame VeriMAP as a solution primarily for *complex, high-stakes tasks* where accuracy is paramount and the cost of failure is high, rather than for simple problems.

**3. Limited Discussion of the Verifier's Role:**
The Verifier module feels somewhat under-described compared to the Planner. It is presented as a relatively simple module that just executes the pre-defined VFs. Given that the NL-based VFs still require an LLM call, the potential for error or ambiguity in these verifications remains. A brief discussion on the reliability of the verifier agent itself, or potential ways to make it more robust, would have been beneficial.

**4. Limitations Section is Strong but Could Look Forward:**
The limitations section is honest and well-written, correctly identifying key challenges like reliance on a strong central planner and resource constraints. However, it could be slightly more forward-looking by suggesting more concrete research directions. For example, what would a "bias-aware verification function" look like? How might one begin to design a "more decentralized" version of VeriMAP?

---

### Summary

**VeriMAP is a novel and impactful contribution to the field of multi-agent LLM systems.** Its key innovation—integrating the generation of specific verification criteria directly into the planning process—proves to be highly effective, leading to state-of-the-art performance on complex reasoning tasks. The paper is supported by exceptionally thorough experimentation and clear presentation. While minor points regarding the planner's inner workings and a more targeted cost-benefit discussion could be improved, these do not detract from the overall quality and significance of the work. It provides a solid foundation for future research in robust and verifiable multi-agent collaboration.

---

# QueST: Incentivizing LLMs to Generate Difficult Problems

Authors: Hanxu Hu, Xingxing Zhang, Jannis Vamvas, Rico Sennrich, Furu Wei

Keywords: Synthetic Data Generation, Difficulty-aware Training, Code Reasoning, Rejection Fine-tuning, Concept Graph, Large Language Models

Comments: 20 pages, 7 figures

Paper link: [http://arxiv.org/abs/2510.17715v1](http://arxiv.org/abs/2510.17715v1)

## Abstract

Large Language Models have achieved strong performance on reasoning tasks, solving competition-level coding and math problems. However, their scalability is limited by human-labeled datasets and the lack of large-scale, challenging coding problem training data. Existing competitive coding datasets contain only thousands to tens of thousands of problems. Previous synthetic data generation methods rely on either augmenting existing instruction datasets or selecting challenging problems from human-labeled data. In this paper, we propose QueST, a novel framework which combines difficulty-aware graph sampling and difficulty-aware rejection fine-tuning that directly optimizes specialized generators to create challenging coding problems. Our trained generators demonstrate superior capability compared to even GPT-4o at creating challenging problems that benefit downstream performance. We leverage QueST to generate large-scale synthetic coding problems, which we then use to distill from strong teacher models with long chain-of-thought or to conduct reinforcement learning for smaller models, proving effective in both scenarios. Our distillation experiments demonstrate significant performance gains. Specifically, after fine-tuning Qwen3-8B-base on 100K difficult problems generated by QueST, we surpass the performance of the original Qwen3-8B on LiveCodeBench. With an additional 112K examples (i.e., 28K human-written problems paired with multiple synthetic solutions), our 8B model matches the performance of the much larger DeepSeek-R1-671B. These findings indicate that generating complex problems via QueST offers an effective and scalable approach to advancing the frontiers of competitive coding and reasoning for large language models.

## Summary

Here is a summary of the paper "QueST: Incentivizing LLMs to Generate Difficult Problems":

**Key Contributions:**
This paper introduces QueST, a novel framework for generating challenging coding problems at scale. The key innovation is training specialized LLM generators to create difficult problems through a combination of difficulty-aware graph sampling and difficulty-aware rejection fine-tuning. This addresses a critical bottleneck in reasoning model development, as current competitive coding datasets are limited to thousands of problems requiring expert human annotation.

**Methods:**
QueST builds on concept graph-based problem generation but enhances it with two difficulty-aware components. First, it uses a novel difficulty estimation metric based on the average majority voting rate of model responses - problems with lower agreement among solutions are considered more difficult. Second, it employs rejection fine-tuning where multiple candidate problems are generated from the same prompt, and only the most difficult one is used for training the generator. Additionally, the concept graph construction incorporates problem difficulty information to bias sampling toward more challenging concept combinations.

**Results:**
The authors generated 100K challenging coding problems, creating the largest synthetic code reasoning dataset to date. When used for distillation, their QueST-8B model trained on 100K synthetic problems plus 112K human-written problems achieved state-of-the-art performance among similar-sized models, closely approaching the performance of the much larger DeepSeek-R1-671B on LiveCodeBench-V5. Ablation studies confirmed that both difficulty-aware components contribute to improved performance, and the generated problems also proved effective for reinforcement learning. The work demonstrates that generating complex problems via QueST offers a scalable approach to advancing reasoning capabilities in language models.

## Critique

Of course. Here is a critique of the paper "QueST: Incentivizing LLMs to Generate Difficult Problems," focusing on its strengths, weaknesses, and overall contribution.

### Summary

This paper introduces **QueST**, a framework designed to train LLMs to generate challenging coding problems. It combines two key innovations: **difficulty-aware graph construction** (to sample complex concept combinations) and **difficulty-aware rejection fine-tuning** (to train the generator to produce harder problems). The authors demonstrate that data generated by their specialized model significantly boosts the performance of smaller student models on competitive coding benchmarks, even rivaling much larger models.

---

### Strengths

1.  **High Novelty and Clear Problem Formulation:** The core idea—training a *specialized generator* for difficult problems rather than relying on prompting a fixed, general-purpose LLM—is highly novel and addresses a critical bottleneck in scaling reasoning models. The paper clearly identifies the limitation of existing human-annotated and synthetically augmented datasets.

2.  **Well-Designed, Multi-Faceted Methodology:** The approach is not a single trick but a cohesive framework.
    *   **Difficulty Proxy (δ):** The proposed difficulty metric, based on the inconsistency of model outputs (average majority voting rate), is intuitive, automated, and empirically validated in Section 3.2. It is a clever way to bypass the need for human-labeled difficulty scores for synthetic data.
    *   **Rejection Fine-Tuning:** This is a powerful and elegant application of a known technique. By always selecting the hardest problem from multiple generations for a given prompt, the method directly optimizes the generator's objective.
    *   **Difficulty-Aware Graph:** Incorporating human-annotated difficulty into the concept graph's edge weights is a simple yet effective way to steer the sampling process toward more challenging concept combinations.

3.  **Significant and Compelling Results:** The experimental results are a major strength.
    *   The distilled 8B model (**QueST-8B**) achieving performance comparable to the 671B DeepSeek-R1 on LiveCodeBench is a striking result that strongly supports the paper's claims.
    *   The experiments are comprehensive, covering data selection analysis, distillation, reinforcement learning, and thorough ablation studies. The results consistently show that QueST-generated data is more beneficial than data from baseline methods.
    *   The "Pareto optimum" claim in Figure 1 is well-supported by the data, showing a better trade-off between model size and performance.

4.  **Clear and Well-Structured Presentation:** The paper is generally easy to follow. The pipeline diagram (Figure 2) effectively illustrates the QueST process. The writing is direct, and the contributions are clearly summarized.

---

### Weaknesses

1.  **Computational Cost and Scalability:** The method is computationally very expensive, a point the authors acknowledge in the limitations. Calculating the difficulty score `δ` for each candidate problem requires:
    *   Generating `T` test inputs.
    *   Generating `M` candidate solutions.
    *   Executing all `M*T` code solutions.
    This process must be repeated `K` times for every prompt during rejection fine-tuning. This cost could be prohibitive for wider adoption and makes real-time RL training infeasible, as noted.

2.  **Limited Exploration of the "Why":** While the paper excellently demonstrates *that* QueST works, it provides less insight into *why* the generated problems are more effective.
    *   Is it purely due to higher difficulty, or is there also an increase in diversity or quality of reasoning concepts?
    *   A deeper analysis of the problem distribution (beyond the top-25 knowledge points) and the structural differences between problems generated by the baseline and QueST would be valuable.

3.  **Narrow Domain Focus:** The work is exclusively evaluated on **coding problems**. While the authors state that other reasoning tasks can be seen as a special case, this remains an assumption. The effectiveness of the difficulty metric `δ` and the overall pipeline for domains like mathematical theorem proving or commonsense reasoning is unproven and non-trivial.

4.  **Ablation Study Could Be Stronger:** The ablation in Table 5 is good but could be more granular. For instance, it would be informative to see the individual contribution of rejection fine-tuning *without* the difficulty-aware graph, and vice versa, in a controlled setting. Table 6 is more of a comparison than a true ablation.

---

### Overall Assessment

This is a **high-quality, impactful paper** with a novel core idea that is backed by strong empirical evidence. The proposed QueST framework represents a significant step forward in synthetic data generation for reasoning tasks. Its main weaknesses are practical (high computational cost) and analytical (a somewhat surface-level exploration of the underlying reasons for its success). The significance of the results, particularly the performance of the distilled 8B model, makes a compelling case for this line of research. It opens up a promising new direction for advancing LLM capabilities by focusing on the data generation process itself.

---

# Enterprise Deep Research: Steerable Multi-Agent Deep Research for Enterprise Analytics

Authors: Akshara Prabhakar, Roshan Ram, Zixiang Chen, Silvio Savarese, Frank Wang, Caiming Xiong, Huan Wang, Weiran Yao

Keywords: Multi-Agent Systems, Enterprise Analytics, Deep Research, Human-in-the-Loop Steering, Autonomous Agents, Knowledge Synthesis, Enterprise AI

Comments: Technical report; 13 pages plus references and appendices

Paper link: [http://arxiv.org/abs/2510.17797v1](http://arxiv.org/abs/2510.17797v1)

## Abstract

As information grows exponentially, enterprises face increasing pressure to transform unstructured data into coherent, actionable insights. While autonomous agents show promise, they often struggle with domain-specific nuances, intent alignment, and enterprise integration. We present Enterprise Deep Research (EDR), a multi-agent system that integrates (1) a Master Planning Agent for adaptive query decomposition, (2) four specialized search agents (General, Academic, GitHub, LinkedIn), (3) an extensible MCP-based tool ecosystem supporting NL2SQL, file analysis, and enterprise workflows, (4) a Visualization Agent for data-driven insights, and (5) a reflection mechanism that detects knowledge gaps and updates research direction with optional human-in-the-loop steering guidance. These components enable automated report generation, real-time streaming, and seamless enterprise deployment, as validated on internal datasets. On open-ended benchmarks including DeepResearch Bench and DeepConsult, EDR outperforms state-of-the-art agentic systems without any human steering. We release the EDR framework and benchmark trajectories to advance research on multi-agent reasoning applications.   Code at https://github.com/SalesforceAIResearch/enterprise-deep-research and Dataset at https://huggingface.co/datasets/Salesforce/EDR-200

## Summary

Based on the provided paper, here is a concise summary focusing on its key contributions, methods, and results.

**Key Contributions:**
This paper introduces Enterprise Deep Research (EDR), a steerable multi-agent system designed for complex, open-ended research tasks in enterprise analytics. Its primary contributions are threefold: (1) a modular, extensible multi-agent architecture configurable for enterprise research; (2) a novel "todo-driven steering framework" that allows for real-time, human-in-the-loop guidance during execution, not just at the initial planning stage; and (3) the release of EDR-200, a dataset of complete research trajectories to advance future work on multi-agent reasoning.

**Methods:**
The EDR framework employs a coordinated system of specialized agents. A central **Master Research Agent** decomposes complex user queries into discrete tasks, managed by a **Research Todo Manager** (`todo.md`), which provides a transparent and modifiable plan visible to both the system and the user. Four specialized search agents (**General Web, Academic, GitHub, LinkedIn**) retrieve information, while domain-specific tools (**NL2SQL, File Analysis, Visualization**) handle enterprise data. The core of its methodology is an iterative **Research Flow Mechanism**. This process involves query decomposition, parallel search execution, result aggregation with deduplication and LLM-driven synthesis, and a critical **reflection phase** that identifies knowledge gaps and updates the task list. The unique **steering integration** allows users to intervene mid-process by issuing natural language directives (e.g., "focus on peer-reviewed sources"), which are queued and processed between iterations to reprioritize or modify the research direction.

**Results:**
EDR was evaluated on several benchmarks against state-of-the-art proprietary and open-source research systems. On **DeepResearch Bench**, it achieved a competitive overall score of 49.86, outperforming most systems and demonstrating particular strength in Instruction-Following and Readability, while using 4x fewer tokens than a leading open-source baseline. On **DeepConsult**, it attained the highest win rate of 71.57% when compared pairwise against OpenAI DeepResearch. On **ResearchQA**, it achieved a coverage of 68.5%, showing strong performance on general, impact, and comparison rubrics, though it highlighted a weakness in citation handling. Internal enterprise evaluations reported a 95% accuracy in SQL generation, a 98% task completion rate, and a 50% reduction in time-to-insight for complex analytical tasks, validating its practical utility.

## Critique

Of course. Here is a detailed critique of the paper "Enterprise Deep Research: Steerable Multi-Agent Deep Research for Enterprise Analytics."

### Summary

This paper presents **Enterprise Deep Research (EDR)**, a multi-agent system designed to perform complex, open-ended research tasks, particularly in enterprise settings. Its key innovations are a modular architecture with specialized agents (for planning, web search, academic search, GitHub, LinkedIn, etc.) and a "steerable" framework that allows for real-time human intervention through a shared `todo.md` file. The system is evaluated on established benchmarks for deep research, where it demonstrates competitive, and in some cases superior, performance compared to both proprietary and open-source alternatives.

---

### Strengths

1.  **Clear Problem Formulation and Motivation:** The paper effectively identifies a significant gap in existing AI research systems: their opacity, rigidity, and lack of integration with enterprise data ecosystems. The argument for "steerability" as a solution to misalignment and inefficiency is well-articulated and compelling.

2.  **Novelty of the "Steerable" Framework:** The core contribution—a real-time, human-in-the-loop steering mechanism—is genuinely novel and practically valuable. Moving beyond pre-execution plan editing to allow dynamic intervention during the research process is a significant step forward for making AI agents more collaborative and trustworthy, especially in high-stakes enterprise environments.

3.  **Comprehensive and Modular System Design:** The architecture is well-thought-out and detailed. The decomposition into a Master Agent, specialized search agents (General, Academic, GitHub, LinkedIn), and domain-specific tools (NL2SQL, Visualization) creates a flexible and extensible framework. The use of the Model Context Protocol (MCP) for enterprise connectors is a pragmatic choice for scalability.

4.  **Rigorous and Extensive Evaluation:** The paper is commendable for its thorough evaluation across three different benchmarks (DeepResearch Bench, DeepConsult, ResearchQA) and internal enterprise use cases. This multi-faceted approach provides a robust picture of the system's capabilities.
    - The results are impressive, showing that EDR is highly competitive with state-of-the-art proprietary systems (like Gemini Deep Research and OpenAI Deep Research) and often outperforms other open-source systems.
    - Including cost and token usage metrics is a practical and valuable addition for researchers and practitioners.

5.  **Significant Contribution to the Community:** Releasing the **EDR-200** dataset of complete research trajectories is a major contribution. This will be an invaluable resource for the research community to study agentic behavior, planning dynamics, and long-horizon reasoning, moving beyond the analysis of just final outputs.

6.  **Clarity of Presentation:** The paper is generally well-structured and clearly written. The use of figures and the detailed breakdown of the research flow mechanism in Section 3 makes the complex system understandable. The appendices with prompts and implementation details add to the paper's reproducibility.

---

### Weaknesses

1.  **Limited Discussion of Limitations:** While the evaluation is extensive, the paper could be more self-critical. The **ResearchQA results reveal a critical weakness**: EDR performs poorly on citation accuracy (CitAcc. in Table 1 is 72.50 vs. 93.37 for the best system) and has "severe weaknesses" in generating examples and handling multi-criteria rubrics. This is a major issue for a system billed as a research assistant, and it deserves a more in-depth discussion of the root causes and potential solutions.

2.  **High Computational Cost:** Despite being more efficient than some baselines, the system is still extremely expensive to run. With an average of ~50 million tokens per run on DeepConsult (costing over $100), its practical applicability is limited to well-funded organizations. The paper does not sufficiently address this as a barrier to widespread adoption.

3.  **"Steerability" is Underexplored in Evaluation:** The central novel feature—human steering—is evaluated with the feature *disabled* ("Real-time steering is kept disabled"). While this is understandable for standardized benchmarking, it leaves a crucial question unanswered: **How much does steering actually improve performance?** A controlled experiment comparing autonomous runs with steered runs would have been a powerful demonstration of the framework's value.

4.  **Ambiguity in Enterprise Use Case Results:** The claims for the internal enterprise evaluation (95% SQL accuracy, 98% task completion, etc.) are strong but lack detail. Without information on the number of tasks, their complexity, or the methodology for measuring "satisfaction" and "time-to-insight," it is difficult to assess the validity and generalizability of these results.

5.  **Clarity of the "Reflection" Mechanism:** While the reflection loop is described as the "core process," the technical details of how the LLM identifies "knowledge gaps," "task misalignment," and "quality inconsistencies" remain somewhat high-level. A more concrete example or a deeper dive into the prompting strategy for reflection would strengthen this section.

### Overall Assessment

This is a **strong and impactful paper**. It presents a novel, well-engineered system that addresses a clear and important problem. The proposed EDR framework represents a significant advance in making multi-agent systems more transparent, controllable, and enterprise-ready. The rigorous benchmarking and the release of the code and trajectory dataset are commendable and will likely spur further research in the field.

The primary weaknesses lie in the under-evaluation of its flagship "steerability" feature and a somewhat superficial treatment of its documented performance limitations in key areas like citation handling. Despite these shortcomings, the paper makes a substantial contribution to the field of AI-powered research and autonomous agents.

---

# ReXMoE: Reusing Experts with Minimal Overhead in Mixture-of-Experts

Authors: Zheyue Tan, Zhiyuan Li, Tao Yuan, Dong Zhou, Weilin Liu, Yueqing Zhuang, Yadong Li, Guowei Niu, Cheng Qin, Zhuyu Yao, Congyi Liu, Haiyang Xu, Boxun Li, Guohao Dai, Bo Zhao, Yu Wang

Keywords: Mixture-of-Experts, Parameter Reusing, Expert Routing, Progressive Scaling, Cross-Layer Reuse

Comments: None

Paper link: [http://arxiv.org/abs/2510.17483v1](http://arxiv.org/abs/2510.17483v1)

## Abstract

Mixture-of-Experts (MoE) architectures have emerged as a promising approach to scale Large Language Models (LLMs). MoE boosts the efficiency by activating a subset of experts per token. Recent works show that fine-grained experts substantially enriches the combinatorial flexibility of active experts and enhances model expressiveness. However, such a design is fundamentally limited by the layer-local routing mechanism: each layer is restricted to its own expert pool. This requires a careful trade-off between expert dimensionality and routing diversity given fixed parameter budgets. We describe ReXMoE, a novel MoE architecture that improves routing beyond the existing layer-local approaches by allowing routers to reuse experts across adjacent layers. ReXMoE decouples expert dimensionality from per-layer budgets, enabling richer expert combinations without sacrificing individual expert capacity or inflating overall parameters. To this end, we propose a new progressive scaling routing (PSR) strategy to gradually increase the candidate expert pool during training. As a result, ReXMoE improves both language modeling and downstream task performance. Extensive experiments on models ranging from 0.5B to 7B parameters across different architectures demonstrate that ReXMoE consistently improves performance under fixed architectural dimensions, confirming ReXMoE as new design paradigm for parameter-efficient and scalable MoE-based LLMs.

## Summary

Based on the provided paper, here is a summary of "ReXMoE: Reusing Experts with Minimal Overhead in Mixture-of-Experts":

**Key Contributions:**
The paper introduces ReXMoE, a novel Mixture-of-Experts (MoE) architecture that addresses the fundamental limitation of layer-local routing in traditional MoE designs. The key innovation is enabling routers to reuse experts across adjacent layers, which significantly expands the candidate expert pool without increasing the total parameter count. Additionally, the authors propose a Progressive Scaling Routing (PSR) strategy that gradually increases the available candidate experts during training to improve stability and performance.

**Methods:**
ReXMoE breaks the layer-local routing constraint by grouping adjacent layers and allowing routers to select experts from this expanded pool. The method only adds negligible router parameters while providing r× more expert combinations (where r is the reuse frequency). The PSR strategy employs curriculum learning by starting with a smaller candidate pool and linearly scaling it up during training, which helps prevent load imbalance issues that commonly occur when dramatically increasing the number of available experts.

**Results:**
Extensive experiments across models ranging from 0.5B to 7B parameters demonstrate that ReXMoE consistently outperforms vanilla MoE baselines on both language modeling perplexity and downstream task performance. The R4 configuration (reusing experts across 4 layers) showed particularly strong results, improving average accuracy on benchmarks by up to 1.08% while reducing perplexity. The method also enables more task-specific expert specialization, as evidenced by qualitative analysis showing distinct expert activation patterns for different tasks. While inference speed sees some degradation in the prefill stage due to increased I/O operations, decoding performance remains comparable to baseline models.

## Critique

Of course. Here is a critique of the paper "ReXMoE: Reusing Experts with Minimal Overhead in Mixture-of-Experts," focusing on its strengths and weaknesses.

### Strengths

1.  **Novel and Well-Motivated Core Idea:** The paper identifies a clear and fundamental limitation in current MoE architectures: the "layer-local" routing mechanism. The proposed solution—reusing experts across adjacent layers to decouple expert dimensionality from per-layer parameter budgets—is elegant, intuitive, and addresses the identified trade-off directly. This is a significant conceptual contribution to MoE design.

2.  **Comprehensive and Rigorous Evaluation:** The experimental setup is thorough. The authors test their method across multiple model scales (0.5B to 7B parameters), different architectures (with and without shared experts), and against strong baselines. The inclusion of both language modeling perplexity and a wide array of downstream task accuracies provides a robust assessment of the method's performance. The comparison to equivalent open-source models further strengthens the claim of effectiveness.

3.  **Practical Contribution with Progressive Scaling Routing (PSR):** The PSR strategy is a crucial and well-designed component that addresses a key training challenge introduced by the larger candidate pool: load imbalance and expert under-utilization. The ablation studies convincingly show that PSR is not just a minor tweak but is essential for unlocking the full potential of the ReXMoE architecture. The comparison between linear and stepwise PSR variants adds depth to the methodology.

4.  **Insightful Analysis:** The paper goes beyond mere performance metrics. The analysis of inference speed impact using vLLM is highly valuable for practitioners. The qualitative analysis of expert activation patterns provides compelling evidence that ReXMoE enables more task-specific specialization, which is a primary goal of the MoE paradigm.

### Weaknesses

1.  **Limited Exploration of Architectural Variants:** The paper primarily focuses on reusing experts within contiguous, grouped layers. It does not explore more flexible or dynamic grouping strategies (e.g., skip-layer connections, or allowing a router to access a non-adjacent "library" of experts). Exploring the upper limits and optimal patterns of expert reuse remains an open question.

2.  **Significant Inference Overhead in Prefill:** While the paper correctly notes that the computational overhead is minimal, it honestly reports a non-negligible performance degradation in the prefill stage due to increased I/O. For certain real-time applications with long prompts, this could be a practical drawback. The paper does not propose optimizations to mitigate this specific issue.

3.  **Clarity and Presentation:** The core idea is clear, but the presentation could be improved in places.
    *   **Figure 1:** The central overview figure is somewhat cluttered and could be simplified to more clearly illustrate the flow of information and the difference between vanilla MoE and ReXMoE.
    *   **Naming Convention:** The model naming (e.g., "ReX-2.3BA0.3B-SE-R4") is descriptive but becomes cumbersome to track throughout the text and tables. A more streamlined notation in the main body would improve readability.
    *   **Methodology Formulation:** The description of the grouping function `G` in Equation 4 could be explained more clearly with a concrete example to ensure the reader understands how layers are grouped, especially at the boundaries.

### Overall Assessment

This is a strong paper that makes a meaningful contribution to the field. The novelty of breaking the layer-local routing constraint is significant, and the proposed method is backed by extensive experiments demonstrating consistent improvements. The inclusion of the PSR training strategy shows a deep understanding of the practical challenges involved. While there are minor weaknesses in presentation and some unexplored architectural avenues, the core contribution is sound, well-validated, and likely to influence future work on efficient and scalable MoE models. The results are significant as they offer a new, parameter-efficient dimension for scaling MoEs without inflating the total parameter count or sacrificing individual expert capacity.

