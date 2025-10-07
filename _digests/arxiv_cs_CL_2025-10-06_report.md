---
title: "ArXiv Daily Digest on 2025-10-06"
collection: digests
type: "ArXiv daily digest"
permalink: /digests/arxiv_cs_CL_2025-10-06_report
date: 2025-10-06
location: "Online"
---

Today's research landscape showcases significant advances in multi-agent collaboration frameworks, with several papers proposing novel approaches to enhance reasoning and coordination. A prominent trend involves leveraging specialized architectures—such as SLM-MUX for orchestrating Small Language Models (SLMs) and frameworks like MARS (Multi-Agent System for Deep ReSearch) that integrate dual-system reasoning—to improve performance on complex tasks like mathematical problem-solving and knowledge-intensive benchmarks. Notably, studies such as LLM-Hanabi highlight the critical role of Theory-of-Mind (ToM) in multi-agent settings, revealing that first-order ToM (interpreting others' intent) correlates more strongly with collaborative success than higher-order reasoning. These developments are complemented by innovative training strategies, including Multi-Agent Tool-Integrated Policy Optimization (MATPO) and methods for mitigating catastrophic forgetting between supervised and reinforcement learning, collectively pushing the boundaries of efficient and scalable AI systems.

## TL;DR

**TL;DR: Multi-Agent Collaboration & Reasoning Efficiency**

Recent papers focus on enhancing AI reasoning through multi-agent collaboration and efficient training strategies:

* **Multi-Agent Systems**: Several works explore orchestrating multiple specialized agents within single models (MATPO, SLM-MUX) or across models, showing significant performance gains over single-agent approaches. Key insight: Avoiding direct text exchanges between small models prevents error propagation while maintaining specialization benefits.

* **Theory of Mind & Collaboration**: LLM-Hanabi demonstrates that first-order Theory-of-Mind (interpreting partner intent) correlates more strongly with collaborative success than higher-order reasoning, providing clear direction for developing collaborative AI.

* **Training Efficiency**: MIFO addresses catastrophic forgetting between supervised and reinforcement learning, achieving SOTA results with only ~1-20% of typical data requirements through selective token training and parameter freezing.

* **Context Engineering**: ACE introduces evolving "playbooks" that accumulate strategies through incremental updates, preventing context collapse and enabling self-improvement without weight updates.

* **Dual-System Reasoning**: MARS combines fast intuitive processing with deliberate reasoning, strategically using external tools while efficiently managing information flow between systems.

Common themes: Moving beyond single-model paradigms, optimizing collaboration dynamics, and developing data-efficient training methods while maintaining or improving reasoning capabilities across mathematical, scientific, and domain-specific tasks.

---

# Slm-mux: Orchestrating small language models for reasoning

Authors: Chenyu Wang, Zishen Wan, Hao Kang, Emma Chen, Zhiqiang Xie, Tushar Krishna, Vijay Janapa Reddi, Yilun Du

Keywords: Multi-Agent Reinforcement Learning, Dual-System Reasoning, Retrieval-Augmented Generation, Tool-Augmented Language Models, Complex Question Answering

Comments: None

Paper link: http://arxiv.org/abs/2510.05077v1

## Abstract

With the rapid development of language models, the number of small language models (SLMs) has grown significantly. Although they do not achieve state-of-the-art accuracy, they are more efficient and often excel at specific tasks. This raises a natural question: can multiple SLMs be orchestrated into a system where each contributes effectively, achieving higher accuracy than any individual model? Existing orchestration methods have primarily targeted frontier models (e.g., GPT-4) and perform suboptimally when applied to SLMs. To address this gap, we propose a three-stage approach for orchestrating SLMs. First, we introduce SLM-MUX, a multi-model architecture that effectively coordinates multiple SLMs. Building on this, we develop two optimization strategies: (i) a model selection search that identifies the most complementary SLMs from a given pool, and (ii) test-time scaling tailored to SLM-MUX. Our approach delivers strong results: Compared to existing orchestration methods, our approach achieves up to 13.4% improvement on MATH, 8.8% on GPQA, and 7.0% on GSM8K. With just two SLMS, SLM-MUX outperforms Qwen 2.5 72B on GPQA and GSM8K, and matches its performance on MATH. We further provide theoretical analyses to substantiate the advantages of our method. In summary, we demonstrate that SLMs can be effectively orchestrated into more accurate and efficient systems through the proposed approach.

## Summary

Here is a concise summary of the paper "MARS: Optimizing Dual-System Deep Research via Multi-Agent Reinforcement Learning":

**Key Contributions:** The paper introduces MARS, a novel multi-agent framework that integrates System 1 (fast, intuitive thinking) and System 2 (deliberate reasoning) within a single LLM to enhance complex reasoning capabilities. Key contributions include: (1) a dual-system collaborative framework for deep research, (2) a multi-agent reinforcement learning approach to optimize both systems simultaneously, (3) a data curation pipeline for diverse training data, and (4) comprehensive evaluation on challenging benchmarks.

**Methods:** MARS establishes a synergistic workflow where System 2 handles deliberate reasoning and strategically invokes external tools (Google Search, Google Scholar, Python Interpreter), while System 1 efficiently processes and distills large volumes of tool outputs using bin-packing optimization. The authors extend Group Relative Policy Optimization (GRPO) to train both systems concurrently, implementing strategies like advantage pre-computation and balanced sampling to prevent either system from dominating learning. This enables efficient collaboration where System 1 filters overwhelming external information, allowing System 2 to focus on reasoning without token consumption trade-offs.

**Results:** Extensive experiments show MARS achieves substantial improvements, including a 3.86% gain on the challenging Humanity's Last Exam (HLE) benchmark and an average 8.9% improvement across 7 knowledge-intensive tasks. Notably, MARS with only 7B parameters outperforms larger models and previous methods, demonstrating the effectiveness of the dual-system paradigm. The framework particularly excels in multi-hop reasoning and technical domains, showing robust performance across diverse complexity levels while efficiently processing large volumes of external information.

## Critique

Of course. Here is a critique of the paper "MARS: Optimizing Dual-System Deep Research via Multi-Agent Reinforcement Learning," evaluating its strengths and weaknesses.

### Strengths

1.  **Novel and Well-Motivated Conceptual Framework:** The core idea of mapping the human cognitive "Dual-System" theory (System 1 for fast, intuitive processing; System 2 for slow, deliberate reasoning) onto a multi-agent LLM framework is highly innovative. It provides a compelling biological and psychological justification for the architecture, addressing a clear problem: the inefficiency of using a single, deliberate reasoning process for all tasks, especially simple ones.

2.  **Technical Sophistication and Integration:** The paper combines several advanced techniques into a cohesive system:
    *   **Multi-Agent RL:** Extending Group Relative Policy Optimization (GRPO) to optimize two "agents" (System 1 and System 2) within the same base model is a non-trivial and technically sound contribution.
    *   **Practical Optimizations:** The use of a bin-packing algorithm (First Fit Decreasing) to manage variable-length tool outputs is a clever and practical solution to a real-world engineering problem, improving computational efficiency.
    *   **Balanced Sampling:** The mechanism to balance the number of training samples between the two systems is crucial for stable multi-agent training and prevents one system from dominating.

3.  **Significant and Extensive Empirical Results:** The results are a key strength of the paper. Achieving state-of-the-art performance on the extremely challenging **Humanity's Last Exam (HLE)** benchmark and a substantial average gain of **8.9%** across seven other knowledge-intensive tasks is a strong validation of the method's effectiveness. The fact that a 7B/8B parameter model can compete with or outperform much larger models and specialized systems is impressive.

4.  **Comprehensive Analysis:** The paper goes beyond just reporting final scores. The analysis section (Section 3.5) provides valuable insights into the training dynamics, tool usage evolution, and the behavior of the two systems, which is essential for understanding how and why the method works. The ablation study (Section 3.6) convincingly demonstrates the contribution of each tool.

### Weaknesses

1.  **Clarity and Presentation of the Core Mechanism:** While the high-level idea is clear, the implementation details of how the two "systems" are distinct within a single LLM could be more explicit. The paper states they are "orchestrated through distinct prompts," but a more detailed explanation or visualization of the specific prompt structures and how the model's behavior is partitioned would enhance clarity. The line between a sophisticated prompt-based workflow and a true architectural innovation feels somewhat blurred.

2.  **Limited Discussion of Limitations:** The paper does not sufficiently address the limitations of the approach.
    *   **Computational Cost:** The system relies on multiple tool calls (Google Search, Scholar), each potentially returning many documents, and multi-turn LLM generation. This incurs significant latency and API costs, which are not discussed.
    *   **Dependence on External Tools:** The performance is heavily dependent on the quality and reliability of external tools (search engines, interpreters). Issues like outdated search indices, paywalled papers, or incorrect code execution are not addressed.
    *   **Generalization:** While tested on knowledge-heavy QA, it's unclear how well this dual-system approach generalizes to other complex tasks like creative writing, strategic planning, or code generation, where the "System 1/System 2" dichotomy might be less applicable.

3.  **Baseline Comparisons and Reproducibility:**
    *   Some baseline results for proprietary models (e.g., OpenAI Deep Research) are listed as "For Reference" without detailed comparison or discussion on the leaderboard, making a direct, fair comparison difficult.
    *   The heavy reliance on proprietary tools (Google Search/Scholar) and the need for a complex RL training pipeline make the system difficult to reproduce for most researchers. A discussion on this and potential simplifications for replication would be helpful.

4.  **Justification for Tool Imbalance:** The analysis shows that Google Search is used in ~98% of tool calls, with Python and Scholar being marginal. While the authors note this is due to the training data, it raises the question of whether the full complexity of a three-tool system is necessary for the achieved performance, or if a simplified system focused on web search would yield similar results.

### Summary

**MARS** presents a **novel, well-executed, and highly effective** framework for complex reasoning tasks. Its strength lies in its compelling bio-inspired concept, sophisticated technical integration, and impressive empirical results that push the state-of-the-art for open-source models. The primary weaknesses are related to the **clarity of its internal mechanics**, an **incomplete discussion of its practical limitations and costs**, and challenges in **reproducibility**. Despite these, the paper makes a significant contribution by demonstrating a powerful new paradigm for structuring language agents to collaborate on deep research problems.

---

# LLM-Hanabi: Evaluating Multi-Agent Gameplays with Theory-of-Mind and Rationale Inference in Imperfect Information Collaboration Game

Authors: Fangzhou Liang, Tianshi Zheng, Chunkit Chan, Yauwai Yim, Yangqiu Song

Keywords: Multi-Agent Reinforcement Learning, Tool-Integrated Planning, Policy Optimization, Language Agents, Multi-Turn Reasoning

Comments: EMNLP 2025 Wordplay

Paper link: http://arxiv.org/abs/2510.04980v1

## Abstract

Effective multi-agent collaboration requires agents to infer the rationale behind others' actions, a capability rooted in Theory-of-Mind (ToM). While recent Large Language Models (LLMs) excel at logical inference, their ability to infer rationale in dynamic, collaborative settings remains under-explored. This study introduces LLM-Hanabi, a novel benchmark that uses the cooperative game Hanabi to evaluate the rationale inference and ToM of LLMs. Our framework features an automated evaluation system that measures both game performance and ToM proficiency. Across a range of models, we find a significant positive correlation between ToM and in-game success. Notably, first-order ToM (interpreting others' intent) correlates more strongly with performance than second-order ToM (predicting others' interpretations). These findings highlight that for effective AI collaboration, the ability to accurately interpret a partner's rationale is more critical than higher-order reasoning. We conclude that prioritizing first-order ToM is a promising direction for enhancing the collaborative capabilities of future models.

## Summary

Here is a summary of the paper "Multi-Agent Tool-Integrated Policy Optimization (MATPO)":

**Key Contributions:** This paper introduces MATPO, a novel multi-agent reinforcement learning framework that enables a single large language model (LLM) to serve as both planner and worker agents through role-specific prompts. The key innovation is enabling effective RL training for multi-agent tool-integrated planning systems within a single model instance, addressing infrastructure challenges of multi-model deployments while preserving specialization benefits.

**Methodology:** MATPO extends single-agent Group Relative Policy Optimization (GRPO) to the multi-agent setting with a principled credit assignment mechanism across planner and worker rollouts. The framework uses a planner-agent for high-level task decomposition and worker-agents for executing specific browsing subtasks, containing noisy tool responses within local contexts. The implementation builds on existing RL frameworks and includes practical components like final-summary mechanisms and user query recapping to improve performance.

**Results:** Experiments on GAIA-text, WebWalkerQA, and FRAMES benchmarks show MATPO consistently outperforms single-agent GRPO baselines by an average of 18.38% relative improvement. The method demonstrates greater robustness to noisy tool outputs and more stable training progress. Ablation studies confirm the importance of key components like final summaries and user query recapping, while also revealing that blocking potentially contaminating URLs has mild effects on performance.

The work provides both theoretical foundations and practical implementation insights for multi-agent RL training, opening directions for scaling to more specialized agents and exploring emergence of new behaviors through multi-role training within single models.

## Critique

Of course. Here is a critique of the paper "Multi-Agent Tool-Integrated Policy Optimization (MATPO)":

### **Strengths**

1.  **High Novelty and Clear Problem Formulation:** The core idea—training a single LLM to perform multiple, distinct agent roles (planner and worker) via reinforcement learning—is highly novel. The paper clearly identifies the limitations of single-agent Tool-Integrated Planning (TIP), such as context window saturation and noise from tool outputs, and proposes a multi-agent framework as a principled solution. The derivation of the MATPO objective from a multi-agent policy gradient provides a solid theoretical foundation.

2.  **Significant and Well-Validated Results:** The experimental results are compelling. An average relative improvement of **18.38%** over the single-agent baseline (GRPO) across three diverse benchmarks (GAIA-text, WebWalkerQA, FRAMES) is a strong result. The demonstration of greater training stability with MATPO, avoiding the performance drops seen in single-agent GRPO, is a significant practical advantage.

3.  **Practical Implementation and Valuable Ablations:** The paper excels in its practical contributions. Figure 4 provides a clear, high-level blueprint for implementing MATPO on top of existing single-agent RL frameworks. The ablation studies are not just performance checks but translate directly into actionable "Practical Take-Aways" for other researchers, such as the necessity of final summaries and the benefits of user query recapping. This greatly enhances the paper's utility and reproducibility.

4.  **Infrastructure Efficiency:** A key advantage rightly highlighted is the "multi-agent-in-one-model" approach. By using a single model instance with role-specific prompts, it avoids the massive memory and orchestration overhead of a "multi-agent-multi-model" system, making the approach far more accessible and scalable.

### **Weaknesses**

1.  **Clarity of the MATPO Objective Formulation:** While the derivation from the policy gradient is sound, the final presentation of the \( J_{\text{MATPO}}(\pi_\theta) \) objective (Section 4.1, Eq. 10-13) could be clearer. The notation, particularly the indexing over \( t \) for different agent rollouts within a full trajectory \( \tau_i \), is dense and may be difficult for readers to parse on a first read. A more verbose explanation or a simplified pseudo-code version of the loss calculation would improve accessibility.

2.  **Limited Exploration of the "One-Model" Dynamics:** The paper convincingly shows that a single model *can* be trained for multiple roles, but it doesn't deeply investigate *how* this works. Does the model learn truly distinct "personas," or is it simply leveraging overlapping capabilities? An analysis of the attention patterns or activation distributions when the model operates under the `planner` vs. `worker` prompt could provide fascinating insights into the inner workings of this multi-role capability.

3.  **Scope of Multi-Agent Complexity:** The experiments are limited to a two-agent system (one planner, one worker). While this is a logical and effective starting point, it leaves open questions about the framework's scalability. The conclusion mentions future work with specialized agents (e.g., for coding), but the current work does not demonstrate this. The performance and stability with 3+ concurrent agent roles remain an empirical question.

4.  **Dependence on High-Quality Prompts:** The method's success is implicitly tied to the quality of the carefully engineered system prompts for the planner and worker agents (provided in the appendix). The paper does not discuss the sensitivity of the results to variations in these prompts. A less optimal prompt design could potentially diminish the observed advantages.

### **Overall Assessment**

This is a **strong and impactful paper**. It introduces a novel and well-motivated algorithm (MATPO) that addresses clear limitations in the current state-of-the-art for agentic LLMs. The significance of the results is high, demonstrated by substantial performance gains and improved training stability. The presentation is generally clear, with excellent figures and highly valuable practical insights, though the core mathematical formulation could be slightly more accessible. The work opens up a promising new direction for efficient and capable multi-agent systems within a single model.

---

# MARS: Optimizing Dual-System Deep Research via Multi-Agent Reinforcement Learning

Authors: Guoxin Chen, Zile Qiao, Wenqing Wang, Donglei Yu, Xuanzhong Chen, Hao Sun, Minpeng Liao, Kai Fan, Yong Jiang, Penguin Xie, Wayne Xin Zhao, Ruihua Song, Fei Huang

Keywords: Agentic Context Engineering, Context Adaptation, Self-Improving Language Models, LLM Agents, Incremental Delta Updates, Grow-and-Refine, Multi-Agent Systems, Context Collapse, Brevity Bias

Comments: Ongoing Work

Paper link: http://arxiv.org/abs/2510.04935v1

## Abstract

Large Reasoning Models (LRMs) often exhibit a tendency for overanalysis in simple tasks, where the models excessively utilize System 2-type, deliberate reasoning, leading to inefficient token generation. Furthermore, these models face challenges in adapting their reasoning capabilities to rapidly changing environments due to the static nature of their pretraining data. To address these issues, advancing Large Language Models (LLMs) for complex reasoning tasks requires innovative approaches that bridge intuitive and deliberate cognitive processes, akin to human cognition's dual-system dynamic. This paper introduces a Multi-Agent System for Deep ReSearch (MARS) enabling seamless integration of System 1's fast, intuitive thinking with System 2's deliberate reasoning within LLMs. MARS strategically integrates multiple external tools, such as Google Search, Google Scholar, and Python Interpreter, to access up-to-date information and execute complex computations, while creating a specialized division of labor where System 1 efficiently processes and summarizes high-volume external information, providing distilled insights that expand System 2's reasoning context without overwhelming its capacity. Furthermore, we propose a multi-agent reinforcement learning framework extending Group Relative Policy Optimization to simultaneously optimize both systems with multi-turn tool interactions, bin-packing optimization, and sample balancing strategies that enhance collaborative efficiency. Extensive experiments demonstrate MARS achieves substantial improvements of 3.86% on the challenging Humanity's Last Exam (HLE) benchmark and an average gain of 8.9% across 7 knowledge-intensive tasks, validating the effectiveness of our dual-system paradigm for complex reasoning in dynamic information environments.

## Summary

Based on the provided paper, here is a summary focusing on its key contributions, methods, and results:

**Summary of "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models"**

This paper introduces ACE (Agentic Context Engineering), a novel framework designed to enhance the performance of Large Language Models (LLMs) by dynamically and continuously adapting their input contexts, rather than updating model weights. The authors identify two key limitations in existing context adaptation methods: (1) **Brevity Bias**, where optimization favors concise but often oversimplified prompts that omit crucial domain-specific details, and (2) **Context Collapse**, where iterative rewriting by an LLM can cause accumulated knowledge to degrade into uninformative summaries. ACE is proposed as a solution to these problems, treating contexts as comprehensive, evolving "playbooks" that accumulate and refine strategies over time.

The core methodological innovation of ACE is its **agentic architecture**, which decomposes the adaptation process into three specialized roles:
1.  **Generator**: Produces reasoning trajectories for new tasks.
2.  **Reflector**: Critiques these trajectories to extract concrete lessons and insights from both successes and failures.
3.  **Curator**: Integrates these distilled insights into the existing context via structured, **incremental delta updates**.

This modular workflow is supported by two key mechanisms:
*   **Incremental Delta Updates**: Instead of costly monolithic context rewrites, ACE makes localized edits by adding or updating small, structured "bullets" of knowledge. This preserves past information and prevents collapse.
*   **Grow-and-Refine**: The framework balances the steady expansion of the context with periodic de-duplication to control redundancy and maintain relevance.

The paper demonstrates ACE's effectiveness through extensive evaluations on two categories of tasks: **agent benchmarks (AppWorld)** and **domain-specific reasoning (financial analysis with FiNER and Formula)**. The key results are:
*   **Performance Gains**: ACE consistently outperformed strong baselines like GEPA, MIPROv2, and Dynamic Cheatsheet, achieving average improvements of **+10.6% on agent tasks** and **+8.6% on financial benchmarks**.
*   **Self-Improvement without Supervision**: A significant result is that ACE achieved strong performance even **without ground-truth labels**, leveraging only natural execution feedback (e.g., code execution success/failure), which is crucial for building autonomous, self-improving agents.
*   **Competitive Leaderboard Performance**: On the challenging AppWorld leaderboard, an agent powered by ACE and a smaller open-source model (DeepSeek-V3.1) matched the average performance of the top-ranked production agent (IBM-CUGA using GPT-4.1) and even surpassed it on the harder test-challenge split.
*   **Efficiency**: ACE was significantly more efficient, reducing adaptation latency by **86.9% on average** and requiring far fewer rollouts and lower token costs compared to other adaptive methods.

In conclusion, ACE presents a powerful and efficient framework for creating self-improving LLM systems by engineering comprehensive and evolving contexts, demonstrating substantial performance gains, robustness without full supervision, and practical efficiency for deployment.

## Critique

Of course. Here is a critique of the paper "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models."

### Strengths

1.  **Novelty and Conceptual Contribution:** The paper introduces a well-defined problem—"context collapse"—and a corresponding solution. The core idea of treating context not as a static or compressed summary but as a structured, evolving "playbook" is a significant conceptual shift. The proposed framework, ACE, with its modular roles (Generator, Reflector, Curator) and its key innovations of **incremental delta updates** and a **grow-and-refine** mechanism, directly addresses the identified limitations of prior work (brevity bias and collapse) in a novel way.

2.  **Comprehensive and Significant Evaluation:** The evaluation is thorough and compelling. The paper demonstrates effectiveness across two distinct and challenging domains:
    *   **Agent Benchmarks (AppWorld):** Showing that ACE enables a smaller open-source model (DeepSeek-V3.1) to match or even surpass a top-ranked proprietary agent (IBM CUGA on GPT-4.1) is a powerful result that highlights the practical significance of the method.
    *   **Domain-Specific Benchmarks (Finance):** Achieving large gains (e.g., +18.0% on Formula) demonstrates that the approach is not limited to agentic tasks but is broadly applicable to knowledge-intensive reasoning.
    The separation of offline and online adaptation scenarios, and the testing with and without ground-truth labels, provides a nuanced view of the method's capabilities and robustness.

3.  **Empirical Rigor and Ablations:** The paper includes a detailed ablation study (Table 3) that convincingly shows the contribution of each key component (Reflector, multi-epoch adaptation, offline warmup). This strengthens the claim that the design choices are intentional and effective. The cost and speed analysis (Table 4) is a major practical strength, demonstrating that ACE is not only more accurate but also significantly more efficient than strong baselines, addressing a critical concern for real-world deployment.

4.  **Clarity of Presentation:** The paper is generally well-written and structured. The motivation is clearly established with the concepts of "brevity bias" and "context collapse," illustrated with a concrete example (Figure 2). The ACE framework is explained step-by-step with the aid of a clear diagram (Figure 4). The results are presented in detailed tables with performance deltas, making comparisons easy.

### Weaknesses

1.  **Dependence on Feedback Quality:** A key limitation, which the authors correctly identify in Section 4.4 and Appendix B, is the method's sensitivity to the quality of feedback signals. The results show that without reliable feedback (e.g., ground-truth labels or clear execution signals), ACE's performance can degrade significantly, even falling below the base model in some cases (e.g., FiNER online without GT). This suggests that ACE's "self-improvement" is conditional on having a high-integrity feedback loop, which may not always be available in real-world applications.

2.  **Complexity and Engineering Overhead:** While the modular design is a strength conceptually, it introduces significant system complexity. Managing three separate LLM-powered components (Generator, Reflector, Curator) with their own prompts and interactions is non-trivial. The paper does not deeply discuss the potential failure modes or orchestration challenges of this multi-agent system, such as what happens if the Reflector itself produces poor-quality insights.

3.  **Limited Exploration of Context Scaling:** The paper rightly notes that modern infrastructure mitigates the cost of long contexts. However, it does not empirically explore the upper limits of this approach. How does performance scale as the context playbook grows to hundreds of thousands or millions of tokens? Is there a point of diminishing returns or information overload for the LLM? A discussion or experiment on the long-term evolution and management of these ever-growing contexts would be valuable.

4.  **Comparison to Stronger Baselines:** While the comparison to GEPA and Dynamic Cheatsheet is strong, it would be even more compelling to see a comparison against other recent advanced prompt optimization or context management techniques, or against a strong fine-tuning baseline (e.g., LoRA) to better position the cost/benefit of context adaptation versus parameter adaptation.

### Overall Assessment

This is a high-quality paper that makes a meaningful contribution to the field of LLM application design. It identifies a clear and important problem, proposes a novel and well-motivated solution, and backs it up with extensive, convincing experiments across multiple domains. The significant performance gains, coupled with the substantial reductions in cost and latency, make a strong case for the practical utility of the ACE framework.

The main weaknesses are related to the practical deployment challenges (feedback dependency, system complexity) rather than the core idea or results. The paper is clear, thorough, and presents a significant step forward in enabling efficient and effective self-improvement for LLM-based systems.

---

# Multi-Agent Tool-Integrated Policy Optimization

Authors: Zhanfeng Mo, Xingxuan Li, Yuntao Chen, Lidong Bing

Keywords: Catastrophic forgetting, Supervised fine-tuning, Reinforcement learning, Parameter freezing, Reasoning enhancement, Data efficiency

Comments: Work in progress

Paper link: http://arxiv.org/abs/2510.04678v1

## Abstract

Large language models (LLMs) increasingly rely on multi-turn tool-integrated planning for knowledge-intensive and complex reasoning tasks. Existing implementations typically rely on a single agent, but they suffer from limited context length and noisy tool responses. A natural solution is to adopt a multi-agent framework with planner- and worker-agents to manage context. However, no existing methods support effective reinforcement learning post-training of tool-integrated multi-agent frameworks. To address this gap, we propose Multi-Agent Tool-Integrated Policy Optimization (MATPO), which enables distinct roles (planner and worker) to be trained within a single LLM instance using role-specific prompts via reinforcement learning. MATPO is derived from a principled credit assignment mechanism across planner and worker rollouts. This design eliminates the need to deploy multiple LLMs, which would be memory-intensive, while preserving the benefits of specialization. Experiments on GAIA-text, WebWalkerQA, and FRAMES show that MATPO consistently outperforms single-agent baselines by an average of 18.38% relative improvement in performance and exhibits greater robustness to noisy tool outputs. Our findings highlight the effectiveness of unifying multiple agent roles within a single LLM and provide practical insights for stable and efficient multi-agent RL training.

## Summary

Here is a summary of the paper "Mitigating Forgetting Between Supervised and Reinforcement Learning Yields Stronger Reasoners":

**Key Contributions:**
The paper introduces MIFO, a plug-and-play framework that addresses the challenge of catastrophic forgetting when combining supervised fine-tuning (SFT) and reinforcement learning (RL) for reasoning tasks in large language models. The authors identify that SFT tends to produce redundant parameter updates with large magnitudes that overwrite the more parsimonious but critical updates learned through RL, leading to performance degradation. MIFO provides an efficient solution that simultaneously tackles data inefficiency, algorithm-specific design limitations, and catastrophic forgetting.

**Methods:**
MIFO employs two key components: data processing and parameter manipulation. For data processing, it dynamically interleaves SFT within RL training by selecting challenging examples based on rollout accuracy and focusing SFT loss calculation only on high-entropy tokens to reduce update magnitude. For parameter manipulation, it maintains a history importance map to identify RL-critical parameters and freezes them during subsequent SFT phases to protect RL-acquired knowledge, then unfreezes them for the next RL step. This approach requires minimal SFT data and remains agnostic to the specific RL or SFT algorithms used.

**Results:**
The method achieves state-of-the-art reasoning performance on mathematical reasoning benchmarks (AIME, AMC, OlympiadBench, MATH500, MMLU-Pro) using only 1.5% of the SFT data and 20.4% of the RL data compared to previous SOTA methods. MIFO also produces more concise reasoning traces with shorter response lengths while maintaining superior performance. Ablation studies confirm that both entropy-based token selection and parameter freezing contribute significantly to mitigating forgetting and improving efficiency. The framework demonstrates robustness across different model sizes (1.5B and 7B parameters) and template variations.

## Critique

Of course. Here is a critique of the paper "Mitigating Forgetting Between Supervised and Reinforcement Learning Yields Stronger Reasoners," focusing on its strengths and weaknesses.

### Overall Summary

This is a strong, well-executed paper that tackles a practical and important problem in the post-training of Large Language Models (LLMs). It presents a clear, plug-and-play framework (MIFO) to address the issue of catastrophic forgetting when interleaving Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL). The empirical results are compelling, and the analysis provides valuable insights into the different update behaviors of SFT and RL.

---

### Strengths

1.  **Clear and Well-Motivated Problem:** The paper identifies a specific, credible, and under-addressed problem: the "SFT forgets RL" phenomenon. The motivation is grounded in a clear empirical observation (SFT updates are larger in magnitude than RL updates) and a compelling hypothesis (this leads to catastrophic forgetting).

2.  **Novel and Insightful Analysis:** The core analysis in Section 3 is a significant strength. The experiments demonstrating that SFT updates are "redundant" while RL updates are "parsimonious" are novel and provide a solid theoretical foundation for the proposed method. This mechanistic insight into the training dynamics is more valuable than simply proposing a new algorithm.

3.  **Well-Designed Method:** MIFO is an elegant solution that directly addresses the identified problem. Its two components are logically derived from the core insights:
    *   **Data Processing:** Using RL rollouts to select challenging examples for SFT and focusing the loss on high-entropy tokens are clever ways to reduce the effective "footprint" of SFT updates.
    *   **Parameter Freezing:** The dynamic identification and freezing of RL-critical parameters is a direct and intuitive way to protect the fragile RL-acquired knowledge from being overwritten.

4.  **Strong and Comprehensive Empirical Results:** The experimental section is thorough.
    *   **State-of-the-Art Performance:** MIFO achieves SOTA or near-SOTA results on a comprehensive suite of challenging mathematical reasoning benchmarks for both 1.5B and 7B models.
    *   **Exceptional Data Efficiency:** The claim of using only **1.5% of SFT data and 20.4% of RL data** compared to the previous SOTA is a standout result with significant practical implications for reducing training costs.
    *   **Response Efficiency:** Producing more concise reasoning traces is a valuable, often-overlooked metric that MIFO also improves.
    *   **Extensive Ablations:** The ablation study in Table 3 clearly demonstrates the contribution of each component (Interleaving, Entropy Selection, Parameter Freezing). The additional analysis in Section 5.4 further validates the method's mechanisms (e.g., showing that high-entropy selection reduces update magnitude).

5.  **"Plug-and-Play" Claim:** The paper effectively argues that MIFO is not tied to a specific RL algorithm (e.g., GRPO), which enhances its generalizability and potential for adoption.

---

### Weaknesses

1.  **Limited Exploration of the "Why":** While the paper excellently demonstrates *that* SFT and RL updates differ, the explanation for *why* remains somewhat surface-level. The theoretical analysis in Appendix C is a good start, but a deeper discussion on the fundamental reasons—perhaps relating to the nature of the loss landscapes, the data distribution (external vs. self-generated), or the optimization objectives—would strengthen the contribution further.

2.  **Scope of Evaluation:** The evaluation is heavily focused on **mathematical reasoning**. While this is a standard and challenging domain, it would be more compelling to see if the "SFT forgets RL" phenomenon and MIFO's effectiveness generalize to other reasoning tasks (e.g., code generation, logical deduction, commonsense QA) or purely linguistic tasks. The inclusion of MMLU-Pro is a step in this direction, but more diversity would bolster the claim of general applicability.

3.  **Complexity and Hyperparameters:** The method introduces several new hyperparameters: the rollout accuracy threshold `p`, the entropy fraction `ρ`, the history coefficient `α`, and the top-k freezing parameter `k`. While a hyperparameter analysis is provided in the appendix, the need to tune these parameters adds complexity to the training recipe. The paper could do more to discuss the sensitivity of the results to these choices or provide practical guidelines for setting them.

4.  **Clarity of Presentation (Minor):**
    *   The paper is generally well-written, but the use of "avg@32" for smaller test sets and "pass@1" for larger ones, while justified, can be slightly confusing at first glance. A more consistent metric would improve readability.
    *   The distinction between `MIFO` and `MIFO†` (with α=0) is crucial, but the naming (`MIFO†` for the simpler variant) is non-standard and could be clearer (e.g., MIFO-static vs. MIFO-momentum).

### Conclusion

This is a high-quality paper with significant contributions. It identifies a real problem, provides a novel analysis of its root cause, and proposes an effective, efficient, and well-evaluated solution. The strengths far outweigh the weaknesses. The novelty lies not just in the MIFO framework itself, but in the foundational analysis of SFT vs. RL update patterns. The results are highly significant due to the simultaneous achievement of SOTA performance, massive data efficiency, and improved output conciseness. The presentation is clear and logically structured, making the work easy to follow and, in principle, reproducible.

---

# Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models

Authors: Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James Zou, Kunle Olukotun

Keywords: Small Language Models, Model Orchestration, Multi-model Systems, Reasoning, Confidence Estimation, Model Selection, Compute Scaling, Test-time Scaling, Discussion-based Methods, Ensemble Methods

Comments: None

Paper link: http://arxiv.org/abs/2510.04618v1

## Abstract

Large language model (LLM) applications such as agents and domain-specific reasoning increasingly rely on context adaptation -- modifying inputs with instructions, strategies, or evidence, rather than weight updates. Prior approaches improve usability but often suffer from brevity bias, which drops domain insights for concise summaries, and from context collapse, where iterative rewriting erodes details over time. Building on the adaptive memory introduced by Dynamic Cheatsheet, we introduce ACE (Agentic Context Engineering), a framework that treats contexts as evolving playbooks that accumulate, refine, and organize strategies through a modular process of generation, reflection, and curation. ACE prevents collapse with structured, incremental updates that preserve detailed knowledge and scale with long-context models. Across agent and domain-specific benchmarks, ACE optimizes contexts both offline (e.g., system prompts) and online (e.g., agent memory), consistently outperforming strong baselines: +10.6% on agents and +8.6% on finance, while significantly reducing adaptation latency and rollout cost. Notably, ACE could adapt effectively without labeled supervision and instead by leveraging natural execution feedback. On the AppWorld leaderboard, ACE matches the top-ranked production-level agent on the overall average and surpasses it on the harder test-challenge split, despite using a smaller open-source model. These results show that comprehensive, evolving contexts enable scalable, efficient, and self-improving LLM systems with low overhead.

## Summary

Here is a summary of the paper "SLM-MUX: Orchestrating Small Language Models for Reasoning":

**Key Contributions:**
This paper identifies a fundamental limitation in existing LLM orchestration methods when applied to Small Language Models (SLMs). While methods like Mixture-of-Agents, LLM-Debate, and Multi-Agent Verification work well for frontier LLMs, they actually harm SLM performance by amplifying errors through "groupthink" during discussion. To address this, the authors propose SLM-MUX, a novel orchestration framework specifically designed for SLMs that avoids explicit text exchanges between models. They also develop complementary optimization strategies including model selection search and compute scaling approaches.

**Methods:**
SLM-MUX operates through a simple yet effective two-phase process: (1) Independent Generation, where each SLM generates multiple candidate responses independently, and (2) Confidence Estimation, where the system selects the final output from the model with the highest self-consistency across its samples. The method uses consistency as a proxy for confidence, breaking ties using validation accuracy. For model selection, the authors propose a search algorithm that maximizes union accuracy while penalizing contradiction cases where overconfident wrong answers suppress correct predictions. They also explore compute scaling along two dimensions: adding more complementary model types and drawing more samples per model.

**Results:**
The method demonstrates significant improvements across multiple reasoning benchmarks. SLM-MUX achieves up to 13.4% improvement on MATH, 8.8% on GPQA, and 7.0% on GSM8K compared to existing orchestration methods. With just two optimally selected SLMs, the framework outperforms Qwen 2.5 72B on GPQA and GSM8K, and matches its performance on MATH. The model selection search consistently identifies complementary model pairs that yield 2.2-4.5% gains over the best single models. The compute scaling analysis reveals that while adding more models has variable effects across benchmarks, increasing samples per model consistently improves performance, with SLM-MUX systematically outperforming baselines like Agent Forest.

## Critique

Of course. Here is a critique of the paper "SLM-MUX: Orchestrating Small Language Models for Reasoning," focusing on its strengths and weaknesses.

### Overall Assessment

This is a strong, well-executed paper that identifies a clear problem, proposes a simple and effective solution, and provides thorough empirical and theoretical validation. The core finding—that orchestrating small language models (SLMs) requires a fundamentally different approach than orchestrating large frontier models—is significant and timely.

---

### Strengths

**1. High Novelty and Clear Problem Identification:**
The paper's greatest strength is its compelling identification of a critical, counter-intuitive problem: **discussion-based orchestration methods, which work for large LLMs, actively harm the performance of SLMs.** This finding challenges a common assumption in the field and opens up a new research direction specifically for efficient, multi-model systems. The analogy to multi-core processors is apt and helps frame the contribution.

**2. Simple, Effective, and Well-Motivated Method:**
The proposed SLM-MUX method is elegant in its simplicity. By avoiding natural language discussion and instead relying on self-consistency as a confidence measure, it sidesteps the "groupthink" problem plaguing SLMs. The method is not just a minor tweak but a principled, alternative architecture for model collaboration. The two-phase process (Independent Generation, Confidence Estimation) is intuitive and easy to understand.

**3. Comprehensive and Convincing Experimental Setup:**
The evaluation is rigorous and leaves little room for doubt:
*   **Clear Baselines:** It compares against all major discussion-based methods (Mixture-of-Agents, LLM-Debate, Multi-Agent Verification) and strong single-model baselines.
*   **Diverse Benchmarks:** Using MATH, GPQA, and GSM8K covers a range of reasoning difficulties and domains.
*   **Scale-Agnostic Analysis:** The direct comparison of the same methods on both SLMs and frontier LLMs (Figure 5) powerfully illustrates the core thesis.
*   **Strong Results:** The performance gains (up to 13.4% on MATH) are substantial. The result that a 2-SLM ensemble can outperform a massive model like Qwen 2.5 72B is a headline-worthy finding that underscores the practical significance of the approach.

**4. Beyond the Core Method: Holistic Framework:**
The paper doesn't stop at SLM-MUX. It thoughtfully addresses the natural follow-up questions:
*   **Model Selection Search:** The formulation of the search objective—maximizing union accuracy while penalizing "overconfident contradictions"—is clever and addresses a key pitfall in naive ensemble selection.
*   **Compute Scaling Analysis:** The systematic exploration of the two scaling dimensions (number of models vs. samples per model) provides valuable practical insights and reveals dataset-dependent behaviors, which is useful for practitioners.

**5. Clear Presentation:**
The paper is well-structured and easy to follow. The use of algorithms, tables, and figures (especially the workflow in Figure 3 and the objective trade-off in Figure 7) effectively complements the text. The writing is generally clear and direct.

---

### Weaknesses and Limitations

**1. Limited Exploration of Confidence Estimation:**
The paper correctly identifies its simple self-consistency metric as a limitation. Relying solely on the frequency of the majority vote is a crude proxy for confidence. A model can be consistently wrong. The paper would be stronger if it included an ablation or discussion of more sophisticated confidence measures (e.g., based on token probabilities or verification scores) even if they were not the final chosen method.

**2. Static and Costly Selection Process:**
The model selection search is exhaustive and performed on a validation set, making it static and computationally expensive for large pools of models. The authors acknowledge this, but it remains a practical limitation. A dynamic, per-instance routing mechanism (e.g., a lightweight router that selects models on the fly) would be a more scalable and powerful future direction.

**3. Scope of "Reasoning" and Generalizability:**
The evaluation is heavily focused on mathematical and scientific reasoning (MATH, GPQA, GSM8K). While these are excellent benchmarks, it's unclear how well SLM-MUX generalizes to other tasks like creative writing, code generation, or open-ended dialogue, where the concept of a single "correct answer" and self-consistency is less defined. A broader evaluation would strengthen the claim of general "reasoning."

**4. Theoretical Analysis Could Be Deeper:**
The mathematical analysis in Section 5 is a good start and provides an intuitive explanation. However, it remains at a high level. A more formal analysis, perhaps framing the problem within a stricter probabilistic ensemble learning framework, could further solidify the theoretical foundations of why SLM-MUX works.

**5. Overhead and Latency:**
While the method is more efficient than using a single massive model, the practical inference overhead of running multiple models (each generating multiple samples) is not deeply discussed. A comparison of total FLOPs or latency against a comparable large model would provide a more complete picture of the efficiency trade-off.

---

### Conclusion

This is an excellent paper that makes a significant contribution. It successfully argues that the paradigm for model orchestration must shift when moving from frontier LLMs to SLMs. The proposed SLM-MUX method is novel, effective, and supported by extensive experiments. While it has some limitations, primarily regarding the sophistication of its confidence estimation and the static nature of model selection, these are clearly acknowledged and provide fertile ground for future work. The paper is likely to influence both research and practice in efficient AI systems.

---

# Mitigating Forgetting Between Supervised and Reinforcement Learning Yields Stronger Reasoners

Authors: Xiangchi Yuan, Xiang Chen, Tong Yu, Dachuan Shi, Can Jin, Wenke Lee, Saayan Mitra

Keywords: Multi-Agent Collaboration, Theory-of-Mind, Rationale Inference, Imperfect Information Games, LLM Evaluation, Cooperative Gameplay

Comments: None

Paper link: http://arxiv.org/abs/2510.04454v1

## Abstract

Large Language Models (LLMs) show strong reasoning abilities, often amplified by Chain-of-Thought (CoT) prompting and reinforcement learning (RL). Although RL algorithms can substantially improve reasoning, they struggle to expand reasoning boundaries because they learn from their own reasoning trajectories rather than acquiring external knowledge. Supervised fine-tuning (SFT) offers complementary benefits but typically requires large-scale data and risks overfitting. Recent attempts to combine SFT and RL face three main challenges: data inefficiency, algorithm-specific designs, and catastrophic forgetting. We propose a plug-and-play framework that dynamically integrates SFT into RL by selecting challenging examples for SFT. This approach reduces SFT data requirements and remains agnostic to the choice of RL or SFT algorithm. To mitigate catastrophic forgetting of RL-acquired skills during SFT, we select high-entropy tokens for loss calculation and freeze parameters identified as critical for RL. Our method achieves state-of-the-art (SoTA) reasoning performance using only 1.5% of the SFT data and 20.4% of the RL data used by prior SoTA, providing an efficient and plug-and-play solution for combining SFT and RL in reasoning post-training.

## Summary

This paper introduces **LLM-Hanabi**, a novel benchmark designed to evaluate multi-agent collaboration, rationale inference, and Theory-of-Mind (ToM) capabilities in large language models (LLMs) using the cooperative card game *Hanabi*. The game is particularly suitable for this purpose due to its imperfect information setting, where players must rely on sparse linguistic hints and infer teammates' intentions to succeed.

The key **contributions** include: (1) the development of an automated evaluation framework that translates game states into natural language, allowing LLM-driven agents to interact and collaborate; (2) a scalable ToM evaluation system that measures both first-order ToM (interpreting a partner's intent) and second-order ToM (predicting how a partner will interpret a hint); and (3) comprehensive benchmarking of a diverse set of LLMs and large reasoning models (LRMs).

The **methods** involve a two-phase evaluation process: during gameplay, agents generate structured reasoning statements (rationale, first-order, and second-order ToM), and post-game, an LLM-as-a-judge scores these statements for alignment. The authors evaluated models in a 5-player Hanabi setup, recording both game performance (score) and ToM proficiency.

**Results** show that LRMs generally outperform standard LLMs in both gameplay and ToM. Notably, there is a strong positive correlation between ToM ability and game success, with first-order ToM (r=0.76) being a more significant predictor of performance than second-order ToM (r=0.58). This suggests that accurately inferring a partner's rationale is more critical for effective collaboration than higher-order reasoning. The findings highlight the importance of prioritizing first-order ToM capabilities in developing future collaborative AI systems.

## Critique

Of course. Here is a critique of the strengths and weaknesses of the paper "LLM-Hanabi: Evaluating Multi-Agent Gameplays with Theory-of-Mind and Rationale Inference".

### Strengths

1.  **Novel and Well-Justified Benchmark:** The paper's core contribution, the LLM-Hanabi benchmark, is a significant and well-designed tool. The choice of Hanabi is excellent, as its cooperative nature and imperfect information mechanics directly isolate and stress-test the specific capabilities of interest: rationale inference and Theory-of-Mind (ToM). This is a clear step up from static, text-based ToM evaluations.

2.  **Clear and Actionable Evaluation Framework:** The methodology for evaluating ToM is a major strength. The breakdown into first-order (interpreting intent) and second-order (predicting interpretation) reasoning is conceptually sound. The automated "LLM-as-a-judge" system for scoring the alignment between generated rationales provides a scalable and quantitative metric that is crucial for large-scale evaluation.

3.  **Significant and Actionable Findings:** The results are not just a leaderboard; they provide a meaningful scientific insight. The strong positive correlation between ToM performance and game success validates the benchmark's premise. More importantly, the finding that **first-order ToM is a stronger predictor of success than second-order ToM** is a nuanced and valuable conclusion. It provides a clear direction for future research, suggesting that improving an agent's ability to *interpret* others may be more impactful than improving its ability to *model* others' interpretations.

4.  **Comprehensive Evaluation:** The paper benchmarks a wide range of models, including both standard LLMs and the newer class of Large Reasoning Models (LRMs). This allows for a rich comparative analysis and highlights the performance gap between the two, adding another layer of insight to the results.

### Weaknesses

1.  **Lack of Baseline Comparison:** A notable weakness is the absence of non-LLM/LRM baselines. The paper does not compare the performance of these agents to simpler rule-based agents, reinforcement learning agents specifically trained for Hanabi, or even human performance. Without these baselines, it is difficult to contextualize the absolute performance levels (e.g., is a score of 30/25 good or bad?) and to fully gauge the significance of the agents' collaborative capabilities.

2.  **Potential Conflation of Abilities:** The benchmark, by design, intertwines several complex abilities: game strategy, planning, communication, and ToM. While the correlation is strong, it is challenging to completely disentangle whether poor performance is due to a failure in ToM or a failure in strategic planning (e.g., knowing *which* card to play even with perfect information). The paper acknowledges the game's complexity as a strength, but a more detailed ablation or error analysis could have strengthened the claim that ToM is the primary bottleneck.

3.  **Reliance on LLM-as-Judge:** The use of an LLM to score ToM statements, while practical, introduces a potential confounder. The quality and potential biases of the judge model can directly affect the reported ToM scores. The paper mentions this in the limitations but does not provide any analysis of the judge's reliability (e.g., through human evaluation of a sample) or its consistency across different judge models.

4.  **Clarity and Presentation:** The presentation is generally clear, but could be improved in a few areas:
    *   **Acronym Introduction:** The term "LRM" (Large Reasoning Model) is used extensively but is not formally defined upon its first appearance, which may confuse some readers.
    *   **Figure Referencing:** The text frequently references Figure 1 and Figure 2, but the figures themselves are not included in the provided text, making it harder to follow the descriptions of the framework and results.
    *   **Metric Detail:** While the ToM scoring process is well-explained, more detail on the exact prompt or rubric given to the "LLM-as-a-judge" would be beneficial for reproducibility.

### Overall Assessment

This is a strong paper that makes a valuable contribution to the field of multi-agent AI and LLM evaluation. The novelty of the benchmark and the importance of its core finding—the primacy of first-order ToM—outweigh its weaknesses. It provides a robust, scalable platform for future research and offers a clear, data-driven recommendation for where to focus efforts in developing more collaborative AI agents. The weaknesses primarily point towards fruitful directions for subsequent work, such as incorporating baselines and conducting a more granular analysis of failure modes.

