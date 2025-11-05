---
title: "ArXiv Daily Digest on 2025-11-04"
collection: digests
permalink: /digests/arxiv_cs_CL_2025-11-04_report
date: 2025-11-04
location: "Online"
---

Today's research landscape reveals a strong emphasis on enhancing multi-agent systems through sophisticated coordination frameworks, with several papers addressing the critical challenge of "lazy agent" behavior—where one agent dominates interactions, undermining collaborative potential. Innovative solutions include reinforcement learning for budget-aware centralized controllers, test-time multimodal reasoning via model orchestration, and causal influence measurements to foster balanced deliberation. Notably, the identified "collaboration gap" highlights that individual model performance does not guarantee effective teamwork, prompting new benchmarks and strategies like "relay inference" to elicit latent collaborative skills. Concurrently, contributions in multilingual NLP introduce PragExTra, the first corpus for pragmatic explicitation, enriching machine translation with cultural adaptability through active learning. These works collectively advance scalable, efficient, and interpretable multi-agent and cross-lingual systems.

## TL;DR

Total papers: 38 , Selected papers: 5

Here's a TL;DR summary of the key themes and insights from these papers:

**Main Theme: Multi-Agent LLM Systems & Collaboration**
Recent research focuses heavily on multi-agent LLM systems, revealing both their potential and fundamental challenges.

**Key Insights:**

1. **Agent Coordination Frameworks** - Papers propose centralized controllers that selectively route queries to expert models (CoRL: https://arxiv.org/abs/2511.02755v1) and master-agent systems for multimodal reasoning (Agent-Omni: https://arxiv.org/abs/2511.02834v1), achieving SOTA performance while controlling costs.

2. **The "Collaboration Gap"** - A critical finding that models performing well individually often fail dramatically when collaborating (https://arxiv.org/abs/2511.02687v1). The "lazy agent" problem emerges where one agent dominates, collapsing multi-agent benefits.

3. **Solutions to Collaboration Issues** - Techniques include causal influence measurement, verifiable rewards, and "relay inference" where stronger agents prime dialogues for weaker ones (Dr. MAMR: https://arxiv.org/abs/2511.02303v1).

4. **Cultural & Pragmatic Adaptation** - PragExTra introduces the first multilingual corpus for pragmatic explicitation in translation, enabling culturally-aware MT systems (https://arxiv.org/abs/2511.02721v1).

**Overall Trend**: The field is moving from single-model approaches to sophisticated multi-agent coordination, with emphasis on cost efficiency, cultural adaptation, and solving fundamental collaboration challenges.

---

# Agent-Omni: Test-Time Multimodal Reasoning via Model Coordination for Understanding Anything

Authors: Huawei Lin, Yunzhi Shi, Tong Geng, Weijie Zhao, Wei Wang, Ravender Pal Singh

Keywords: Multi-agent systems, Multimodal reasoning, Model coordination, Test-time inference, Omni-modal understanding

Comments: 16 pages, 7 figures, 14 tables. Under Review

Paper link: [http://arxiv.org/abs/2511.02834v1](http://arxiv.org/abs/2511.02834v1)

## Abstract

Multimodal large language models (MLLMs) have shown strong capabilities but remain limited to fixed modality pairs and require costly fine-tuning with large aligned datasets. Building fully omni-capable models that can integrate text, images, audio, and video remains impractical and lacks robust reasoning support. In this paper, we propose an Agent-Omni framework that coordinates existing foundation models through a master-agent system, enabling flexible multimodal reasoning without retraining. The master agent interprets user intent, delegates subtasks to modality-specific agents, and integrates their outputs into coherent responses. Extensive experiments across text, image, audio, video, and omni benchmarks show that Agent-Omni consistently achieves state-of-the-art performance, particularly on tasks requiring complex cross-modal reasoning. Its agent-based design enables seamless integration of specialized foundation models, ensuring adaptability to diverse inputs while maintaining transparency and interpretability. In addition, the framework is modular and easily extensible, allowing future improvements as stronger models become available. %We release an open-source implementation to support continued research on scalable and reliable omni-modal reasoning.

## Summary

Here is a summary of the paper "Agent-Omni: Test-Time Multimodal Reasoning via Model Coordination for Understanding Anything":

**Key Contributions:**
This paper introduces Agent-Omni, a novel framework that enables comprehensive multimodal reasoning by coordinating existing foundation models through a master-agent system, without requiring any task-specific fine-tuning or retraining. The main contribution is a training-free approach that can flexibly handle arbitrary combinations of text, images, audio, and video inputs to produce coherent textual outputs.

**Methods:**
Agent-Omni employs a hierarchical architecture where a Master Agent serves as the central controller with four key stages: Perception (analyzes input modalities into structured representations), Reasoning (decomposes user queries into modality-specific sub-questions), Execution (invokes appropriate foundation models from a model pool), and Decision (integrates outputs and determines if iterative refinement is needed). The framework coordinates specialized foundation models like Deepseek R1 for text, Claude 3.7 Sonnet for image/video, and Qwen2.5 Omni for audio through an iterative self-improvement loop that enables progressive refinement of answers.

**Results:**
Extensive experiments across text (MMLU, MMLU-Pro), image (MathVision, MMMU), video (VideoMathQA, STI-Bench), audio (MMAU, MELD-Emotion), and omni benchmarks (Daily-Omni, OmniBench) show that Agent-Omni consistently achieves state-of-the-art performance. It particularly excels on complex cross-modal reasoning tasks, outperforming both individual foundation models and DSPy-CoT baselines. The framework demonstrates robust performance across modalities while maintaining transparency and interpretability, though it introduces higher latency due to model coordination (4-7s on unimodal tasks, up to 20s on video tasks). The ablation studies confirm that iterative reasoning provides incremental accuracy gains, with most queries resolving in the first iteration while complex ones benefit from additional refinement cycles.

## Critique

Based on the provided paper, here is an analysis of its strengths and weaknesses:

**Strengths:**

*   **High Novelty and Practical Approach:** The core idea of "Agent-Omni"—orchestrating existing, specialized foundation models at test-time to achieve "omni-modal" understanding without any joint fine-tuning—is highly novel and addresses a significant practical bottleneck. It circumvents the immense cost and data requirements of training a single, unified omni-model, presenting a flexible and resource-efficient alternative.
*   **Strong Empirical Validation:** The paper provides extensive evaluations across a wide range of modalities (text, image, video, audio) and challenging benchmarks (MMLU-Pro, MMMU-Pro, etc.). The results are compelling, showing that Agent-Omni not only matches but often surpasses the performance of state-of-the-art foundation models and a strong baseline (DSPy-CoT), particularly on complex omni-modal and video tasks. This strongly supports the paper's claims.
*   **Clear and Well-Structured Presentation:** The paper is well-organized. The "Master Agent" workflow (Perception -> Reasoning -> Execution -> Decision) is explained clearly with a helpful overarching example (the accident scenario). The use of tables and figures effectively summarizes model capabilities, results, and the framework's architecture.
*   **Comprehensive Ablation Studies:** The paper includes meaningful ablation studies that investigate the impact of the number of reasoning iterations and the choice of foundation models. The "exit rate" analysis is particularly insightful, demonstrating that the framework is efficient for simple queries while leveraging its iterative loop for complex ones. This adds depth to the results.

**Weaknesses:**

*   **Limited Technical Depth on Core Mechanism:** While the high-level workflow is clear, the paper lacks detail on the *precise mechanisms* that enable effective cross-modal reasoning and integration. For instance, how does the "Decision" stage concretely resolve inconsistencies between conflicting outputs from different modality-specific agents? The prompts and JSON schemas are relegated to an appendix; a brief discussion of their key design principles in the main text would strengthen the methodology section.
*   **Insufficient Analysis of Latency/Compute Trade-off:** The paper openly reports that Agent-Omni incurs significantly higher latency (up to 20x slower on some tasks) compared to single foundation models. However, the discussion of this trade-off is relatively brief. A more nuanced analysis—perhaps breaking down the latency by component (e.g., master agent reasoning vs. model pool execution) and discussing potential optimizations (like parallel execution)—would be valuable for assessing the framework's practicality.
*   **Superficial Treatment of Limitations:** The "Limitations" section is quite generic (e.g., "relies on external models," "may propagate biases"). It misses the opportunity to discuss limitations specific to the *orchestration approach*, such as: How does the system's performance degrade with a less capable "Master Agent"? What happens if the model pool lacks a specialist for a specific sub-task? Is there a risk of the iterative loop leading to "reasoning loops" or over-complication on simple tasks?
*   **Vague Claims on "Interpretability":** The paper claims the framework ensures "transparency and interpretability," but this is not substantiated. While the structured JSON outputs provide a trace, it's not demonstrated how this translates into a genuinely interpretable or debuggable system for a human. A small case study illustrating this would make the claim more convincing.

**Summary:**

This is a strong paper that presents a novel, effective, and well-evaluated framework for a challenging problem. Its primary strength lies in its compelling core idea and the extensive empirical evidence supporting it. The main weaknesses are a lack of deep technical detail on the integration mechanics and a somewhat superficial discussion of the inherent latency trade-offs and specific limitations of the agent-coordination paradigm. Despite these points, the significance of the results and the clarity of the high-level presentation make a convincing case for the value of the proposed approach.

---

# Unlocking the Power of Multi-Agent LLM for Reasoning: From Lazy Agents to Deliberation

Authors: Zhiwei Zhang, Xiaomin Li, Yudi Lin, Hui Liu, Ramraj Chandradevan, Linlin Wu, Minhua Lin, Fali Wang, Xianfeng Tang, Qi He, Suhang Wang

Keywords: Multi-Agent LLM Reasoning, Lazy Agent Problem, Causal Influence, Reinforcement Learning, Meta-Reasoning

Comments: None

Paper link: [http://arxiv.org/abs/2511.02303v1](http://arxiv.org/abs/2511.02303v1)

## Abstract

Large Language Models (LLMs) trained with reinforcement learning and verifiable rewards have achieved strong results on complex reasoning tasks. Recent work extends this paradigm to a multi-agent setting, where a meta-thinking agent proposes plans and monitors progress while a reasoning agent executes subtasks through sequential conversational turns. Despite promising performance, we identify a critical limitation: lazy agent behavior, in which one agent dominates while the other contributes little, undermining collaboration and collapsing the setup to an ineffective single agent. In this paper, we first provide a theoretical analysis showing why lazy behavior naturally arises in multi-agent reasoning. We then introduce a stable and efficient method for measuring causal influence, helping mitigate this issue. Finally, as collaboration intensifies, the reasoning agent risks getting lost in multi-turn interactions and trapped by previous noisy responses. To counter this, we propose a verifiable reward mechanism that encourages deliberation by allowing the reasoning agent to discard noisy outputs, consolidate instructions, and restart its reasoning process when necessary. Extensive experiments demonstrate that our framework alleviates lazy agent behavior and unlocks the full potential of multi-agent framework for complex reasoning tasks.

## Summary

This paper addresses a critical limitation in multi-agent large language model (LLM) reasoning frameworks: the emergence of "lazy agent" behavior where one agent dominates while the other contributes minimally, effectively collapsing the multi-agent system into an ineffective single-agent setup.

The key contributions are threefold. First, the authors provide a theoretical analysis revealing that the normalization term in multi-turn Group Relative Preference Optimization (GRPO) objectives inherently biases models toward trajectories with fewer turns, incentivizing agents to bypass collaborative interactions and leading to lazy behavior. Second, they propose a Shapley-inspired causal influence measurement that groups semantically similar steps across rollouts to provide stable estimates of each agent's contribution, mitigating wording bias without requiring costly resampling. Third, they design a verifiable reward mechanism that enables the reasoning agent to discard noisy intermediate outputs and restart reasoning when necessary, preventing error propagation in extended multi-turn interactions.

The proposed method, Dr. MAMR (Multi-agent Meta-Reasoning Done Right), integrates these components by removing the problematic normalization, incorporating causal influence signals, and adding restart rewards into the advantage function. Extensive experiments across seven mathematical reasoning benchmarks show that Dr. MAMR consistently outperforms both single-agent GRPO and the multi-agent ReMA baseline, with performance gains increasing with model scale. The method also demonstrates superior training stability and maintains balanced agent contributions throughout training, unlike ReMA where the reasoning agent's influence diminishes to near-zero. Ablation studies confirm that all three components contribute to the overall performance improvement.

## Critique

Of course. Here is a critique of the paper "Unlocking the Power of Multi-Agent LLM for Reasoning: From Lazy Agents to Deliberation."

### Strengths

1.  **Novelty and Problem Identification:** The paper's greatest strength is identifying and deeply analyzing a critical, yet previously underexplored, problem in multi-agent LLM systems: the "lazy agent" phenomenon. The observation that one agent can become dominant even in a *sequential* setting, effectively collapsing the multi-agent system into a single-agent one, is both surprising and significant. This moves beyond simply proposing a new architecture and instead focuses on a fundamental flaw in existing training paradigms.

2.  **Comprehensive Solution:** The proposed Dr. MAMR framework is not a single trick but a multi-faceted solution addressing different aspects of the problem. The three core components are well-motivated:
    *   **Theoretical Analysis:** Providing a formal theorem explaining why the normalization term in multi-turn GRPO inherently biases the system towards lazy behavior is a high-quality contribution. It roots the empirical observation in a solid theoretical foundation.
    *   **Shapley-inspired Causal Influence:** This is a clever and practical method to estimate an agent's contribution without the prohibitive cost of full Shapley value calculation. The idea of grouping semantically similar steps across rollouts is innovative and efficiently addresses the issue of phrasing bias.
    *   **Restart Mechanism with Verifiable Reward:** This component proactively addresses a secondary problem (error propagation in long interactions) that becomes more relevant once the primary lazy-agent issue is solved. The design of a verifiable reward for the restart action is elegant and ensures the behavior is learned, not just prompted.

3.  **Rigorous and Extensive Evaluation:** The experimental section is thorough. The authors don't just show final scores; they provide compelling evidence through:
    *   **Causal Influence Tracking:** Demonstrating how the influence of each agent evolves during training for both ReMA and Dr. MAMR powerfully validates their core claim.
    *   **Ablation Studies:** Clearly showing the contribution of each component (Normalization Debias, Causal Influence, Restart Behavior).
    *   **Multiple Benchmarks and Model Sizes:** Testing on seven diverse mathematical reasoning benchmarks and across 3B, 7B, and 14B model scales strengthens the generalizability of the results.
    *   **Training Stability:** Showing that Dr. MAMR leads to more stable training than ReMA is a crucial practical benefit.

4.  **Clarity of Presentation:** The paper is generally well-written. The problem is introduced clearly, the methodology is explained step-by-step with supporting theory and preliminary experiments, and the results are presented logically. The figures, particularly the causal influence and training curve plots, are effective at conveying key insights.

### Weaknesses

1.  **Computational Overhead:** While the Shapley-inspired method is more efficient than a full calculation, it still introduces non-trivial computational overhead. The paper would benefit from a discussion of the training cost (e.g., time or FLOPs) compared to the ReMA baseline. Grouping steps by semantic similarity requires running an encoder model, which adds complexity.

2.  **Scope of Evaluation:** The evaluation is heavily focused on **mathematical reasoning**. While this is a standard and challenging domain, it remains to be seen how well the identified "lazy agent" problem and the Dr. MAMR solution generalize to other complex reasoning tasks, such as strategic planning, commonsense reasoning, or long-horizon dialogue. The restart mechanism, in particular, might have different dynamics in non-mathematical contexts.

3.  **Hyperparameter Sensitivity:** The method introduces new hyperparameters (e.g., `α` and `β` for combining advantages in Eq. 8, thresholds for semantic similarity). The paper does not discuss the sensitivity of the results to these choices or the process for tuning them, which is important for reproducibility and practical adoption.

4.  **Clarity on "Lazy" vs. "Efficient" Behavior:** The paper could more precisely define the boundary between a "lazy" action (e.g., empty output, simple summarization) and a genuinely efficient action that correctly concludes a reasoning step early. The theoretical analysis correctly points out the bias against longer trajectories, but a more nuanced discussion of what constitutes beneficial vs. detrimental shortcut-taking would be valuable.

### Overall Assessment

This is a **high-quality paper** with significant contributions. It successfully identifies a critical, non-obvious failure mode in a promising area (multi-agent LLMs), provides a deep theoretical and empirical analysis of its cause, and proposes a novel, multi-component framework that effectively addresses it. The results are compelling, demonstrating that fixing the lazy-agent problem allows multi-agent systems to consistently outperform their single-agent counterparts.

The main limitations are the confined evaluation domain and the lack of discussion on computational cost. However, the core ideas—diagnosing training objective biases, robust causal influence estimation, and verifiable restart mechanisms—are likely to be influential and inspire future work in multi-agent and long-horizon RL for LLMs. The paper is a strong step towards unlocking the true collaborative potential of LLM agents.

---

# The Collaboration Gap

Authors: Tim R. Davidson, Adam Fourney, Saleema Amershi, Robert West, Eric Horvitz, Ece Kamar

Keywords: multi-agent collaboration, collaboration gap, maze-solving benchmark, heterogeneous agents, relay inference, AI-AI interaction

Comments: None

Paper link: [http://arxiv.org/abs/2511.02687v1](http://arxiv.org/abs/2511.02687v1)

## Abstract

The trajectory of AI development suggests that we will increasingly rely on agent-based systems composed of independently developed agents with different information, privileges, and tools. The success of these systems will critically depend on effective collaboration among these heterogeneous agents, even under partial observability. Despite intense interest, few empirical studies have evaluated such agent-agent collaboration at scale. We propose a collaborative maze-solving benchmark that (i) isolates collaborative capabilities, (ii) modulates problem complexity, (iii) enables scalable automated grading, and (iv) imposes no output-format constraints, preserving ecological plausibility. Using this framework, we evaluate 32 leading open- and closed-source models in solo, homogeneous, and heterogeneous pairings. Our results reveal a "collaboration gap": models that perform well solo often degrade substantially when required to collaborate. Collaboration can break down dramatically; for instance, small distilled models that solve mazes well alone may fail almost completely in certain pairings. We find that starting with the stronger agent often improves outcomes, motivating a "relay inference" approach where the stronger agent leads before handing off to the weaker one, closing much of the gap. Our findings argue for (1) collaboration-aware evaluation, (2) training strategies developed to enhance collaborative capabilities, and (3) interaction design that reliably elicits agents' latent skills, guidance that applies to AI-AI and human-AI collaboration.

## Summary

Based on the provided paper, here is a summary of its key contributions, methods, and results.

### Key Contributions
This paper introduces and provides empirical evidence for the "collaboration gap," a phenomenon where AI agents that perform well individually suffer a significant performance drop when required to collaborate with each other. The authors develop a novel benchmark for collaborative maze-solving that isolates collaborative capabilities, allows for scalable automated grading, and imposes minimal output constraints to maintain ecological plausibility. A key methodological contribution is the proposal of "relay inference," a strategy where a stronger agent primes the initial interaction before handing off to a weaker partner, which is shown to substantially improve collaborative outcomes.

### Methods
The core methodology involves a collaborative maze-solving task where two agents, each with a partially obfuscated view of a 6x6 maze, must communicate via natural language to agree on a path from the start to the goal. The study evaluates 32 leading open- and closed-source models across three settings: 1) **Solo baselines** (with full and distributed maze information), 2) **Homogeneous collaboration** (two identical agents working together), and 3) **Heterogeneous collaboration** (agents from different model families or of varying capabilities working together). Performance is measured using a weighted outcome metric and binary success rate, with a third "grader" agent automatically extracting and evaluating the proposed solution from the dialogue transcript. The "relay inference" technique is tested by having a strong and weak agent interact for a set number of turns before swapping in another copy of the weak agent (or vice-versa) to complete the task.

### Key Results
The results reveal a substantial collaboration gap: most models, including high-performing ones, experienced a significant performance drop when collaborating with a copy of themselves compared to solving the maze solo. This gap was especially pronounced in smaller, distilled models. The study also found strong **ordering effects** in heterogeneous collaborations; which agent initiates the dialogue heavily influences the outcome, with starting with the stronger agent generally leading to better performance. Furthermore, the proposed **relay inference** strategy was highly effective; priming a dialogue with just the first message from a strong agent dramatically boosted the performance of weaker collaborators. Conversely, having a strong agent attempt to "recover" a dialogue after several turns of weak-weak interaction was far less effective. These findings suggest that effective collaboration is a distinct capability not adequately captured by current model training paradigms, highlighting the need for collaboration-aware evaluation and training strategies.

## Critique

Of course. Here is a critique of the strengths and weaknesses of the paper "The Collaboration Gap."

### Overall Summary

This is a timely, well-executed, and significant paper that identifies a critical, under-explored failure mode in modern language models: a "collaboration gap" where models perform well alone but fail dramatically when required to collaborate with another agent. The work is notable for its scale, the cleverness of its benchmark, and the actionable insights it provides.

---

### Strengths

**1. High Novelty and Timeliness:**
*   **Novel Problem:** The paper tackles a crucial, emerging problem in AI: how independently developed agents collaborate in the wild. While multi-agent systems are a hot topic, this paper uniquely focuses on *emergent, unstructured collaboration* without a central orchestrator, which is a more realistic and challenging scenario.
*   **Novel Benchmark:** The "collaborative maze-solving" task is an elegant and ingenious choice. It effectively isolates collaborative capabilities like grounding, negotiation, and conflict resolution, while being automatically gradable and scalable. The "distributed information" twist is a simple but powerful mechanic to force collaboration.

**2. Significant and Surprising Results:**
*   **The Core Finding:** The existence of a large "collaboration gap" is a powerful and somewhat counterintuitive result. It compellingly argues that collaboration is a distinct capability not automatically acquired through standard training.
*   **Actionable Insights:** The paper goes beyond just identifying a problem. The discovery of "ordering effects" and the subsequent proposal of "relay inference" is a highly practical contribution. Showing that a strong model can "prime" a dialogue for a weaker one, even for just a few turns, is a valuable strategy for efficient system design.
*   **Scale and Breadth:** Evaluating 32 models across solo, homogeneous, and heterogeneous settings provides a comprehensive and convincing landscape of the current state of AI-AI collaboration. The finding that distilled models are disproportionately affected is particularly insightful.

**3. Clarity and Rigor:**
*   **Excellent Presentation:** The paper is exceptionally well-written and structured. The figures are clear and directly support the narrative. The use of qualitative dialogue snippets (e.g., Figures 4.2, 4.4, 4.5) is highly effective in illustrating the abstract concepts of "grounding" and "style imitation."
*   **Thorough Evaluation:** The authors demonstrate commendable rigor. They include extensive appendices covering grading consistency, hyperparameter ablations, and error analysis. The discussion of non-determinism and reproducibility is honest and strengthens the credibility of the results.

---

### Weaknesses

**1. Ecological Validity and Task Simplicity:**
*   **The "Lower Bound" Argument:** While the authors rightly frame mazes as a "lower bound," the simplicity of the task is the paper's primary limitation. Real-world collaborations involve ambiguity, deception, complex tool use, and long-term planning horizons that are absent here. The gap observed in mazes likely underestimates the challenges in more complex domains. A discussion of how these findings might extrapolate to richer environments (e.g., Overcooked, Minecraft) would have been beneficial.

**2. Limited Analysis of *Why* the Gap Exists:**
*   **Surface-Level Diagnosis:** The paper excellently *demonstrates* the gap and its symptoms (poor grounding, imitation) but provides a relatively shallow analysis of the root causes. Is the gap due to:
    *   A lack of specific training data featuring peer-to-peer collaboration?
    *   The models' inherent tendency to be sycophantic or imitate the style of their partner?
    *   A fundamental inability for auto-regressive models to maintain a consistent, shared mental state over a long dialogue?
    A deeper mechanistic analysis or hypotheses about the architectural or training data origins of this problem would have strengthened the paper further.

**3. Potential Confounding Factors:**
*   **Prompt Sensitivity:** The entire experiment rests on a single set of prompts. While the authors address this in the reproducibility statement, the results could be sensitive to the specific instructions given. A small ablation showing how performance changes with different prompting strategies (e.g., explicitly instructing agents to establish a coordinate system first) would have been informative.
*   **Cost of Non-Determinism:** The acknowledgment that non-determinism "explodes" in multi-turn interactions is a valid concern. While they mitigate it with large sample sizes, it remains a source of noise that is harder to control than in standard single-model evaluations.

### Conclusion

This is a standout paper that makes a clear and important contribution. It identifies a critical, previously underexplored bottleneck in the path towards a multi-agent future. The strengths of its novel benchmark, significant results, and clear presentation far outweigh its weaknesses. It successfully sounds an alarm bell for the community and provides a solid foundation and a new benchmark for future work aimed at "designing in" collaborative capabilities from the start.

---

# PragExTra: A Multilingual Corpus of Pragmatic Explicitation in Translation

Authors: Doreen Osmelak, Koel Dutta Chowdhury, Uliana Sentsova, Cristina España-Bonet, Josef van Genabith

Keywords: pragmatic explicitation, multilingual corpus, translation studies, cultural adaptation, active learning, machine translation

Comments: None

Paper link: [http://arxiv.org/abs/2511.02721v1](http://arxiv.org/abs/2511.02721v1)

## Abstract

Translators often enrich texts with background details that make implicit cultural meanings explicit for new audiences. This phenomenon, known as pragmatic explicitation, has been widely discussed in translation theory but rarely modeled computationally. We introduce PragExTra, the first multilingual corpus and detection framework for pragmatic explicitation. The corpus covers eight language pairs from TED-Multi and Europarl and includes additions such as entity descriptions, measurement conversions, and translator remarks. We identify candidate explicitation cases through null alignments and refined using active learning with human annotation. Our results show that entity and system-level explicitations are most frequent, and that active learning improves classifier accuracy by 7-8 percentage points, achieving up to 0.88 accuracy and 0.82 F1 across languages. PragExTra establishes pragmatic explicitation as a measurable, cross-linguistic phenomenon and takes a step towards building culturally aware machine translation. Keywords: translation, multilingualism, explicitation

## Summary

This paper introduces **PragExTra**, the first multilingual corpus and detection framework for **pragmatic explicitation** in translation. Pragmatic explicitation refers to the phenomenon where translators enrich texts with background details—such as entity descriptions, measurement conversions, or cultural clarifications—to make implicit cultural or contextual knowledge explicit for target audiences. While widely acknowledged in translation studies, this phenomenon had not been systematically modeled or measured computationally prior to this work.

The authors constructed the PragExTra corpus using parallel data from **TED-Multi** and **Europarl**, covering ten language pairs. They identified candidate explicitation instances through **null alignments**—words appearing only in the target translation—and refined them using linguistic heuristics (e.g., named entity recognition and POS filtering). An **active learning framework** was employed to iteratively improve detection, combining uncertainty sampling and diversity-based querying with human annotation. The final corpus includes 2,700 annotated sentence pairs, categorized into four main types of explicitations:
- **Entities (ENT)**: e.g., adding descriptions or replacing named entities
- **System Transfers (SYS)**: e.g., converting measurement units or educational systems  
- **Linguistic Adjustments (LING)**: e.g., translator remarks or acronym expansions
- **Added Information (ADD)**: e.g., contextual clarifications or deixis resolution

Key results demonstrate that:
- The active learning framework significantly improved classifier performance, boosting accuracy by **77–88 percentage points** and achieving up to **0.88 accuracy** and **0.82 F1-score** across languages.
- Entity and system-level explicitations were the most frequent, reflecting translators' efforts to bridge cultural and institutional knowledge gaps.
- Cross-lingual evaluation showed that typologically similar languages (e.g., Portuguese to Spanish, Dutch to German) achieved stronger transfer performance, while more distant languages (e.g., Greek) saw more modest gains.

PragExTra establishes pragmatic explicitation as a measurable, cross-linguistic phenomenon and provides a valuable resource for developing culturally aware machine translation systems. The authors plan to expand the corpus to more languages and explore applications in improving cross-cultural NLP systems.

## Critique

Of course. Here is a critique of the paper "PragExTra: A Multilingual Corpus of Pragmatic Explicitation in Translation," focusing on its strengths, weaknesses, novelty, significance, and clarity.

### Overall Impression

This is a strong and valuable contribution to the fields of computational linguistics and machine translation. The paper successfully identifies and formalizes a long-discussed but computationally under-explored phenomenon—pragmatic explicitation—and provides a high-quality, human-validated resource for its study.

---

### Strengths

1.  **High Novelty and Clear Problem Formulation:** The core strength of the paper is its novelty. It moves beyond well-studied structural/discourse explicitations (like adding connectives) to focus on the culturally and pragmatically motivated additions that are crucial for real-world translation quality. The formalization in Section 3 provides a clear, actionable taxonomy (Entities, System Transfers, Linguistic Adjustments, Added Information) that makes this abstract concept computationally tractable.

2.  **Robust and Resource-Efficient Methodology:** The combination of automatic candidate extraction (using null alignments and linguistic filters) with an **active learning framework** is a well-justified and practical approach. Given the inherent sparsity of the phenomenon (<1% of tokens), this human-in-the-loop strategy is not just a bonus but a necessity for building a quality corpus without prohibitive cost. The use of multiple query strategies (diversity-based and uncertainty-based) is sophisticated and appropriate.

3.  **Significant and Actionable Results:** The results are compelling and directly support the paper's claims.
    *   The performance gains from active learning (7-8% improvement in accuracy/F1) are substantial and empirically validate the chosen methodology.
    *   The cross-lingual evaluation demonstrates that the phenomenon and the detection model generalize well, especially to typologically similar languages. The observed typological patterns (e.g., Germanic languages adding more dimensional information) are a fascinating finding that opens doors for further linguistic research.
    *   The creation and release of the **PragExTra corpus itself** is a significant contribution, providing a benchmark for future work on culturally-aware NLP.

4.  **Excellent Presentation and Examples:** The paper is generally well-written and easy to follow. The use of concrete, illustrative examples throughout the paper (especially in Table 3) is highly effective. It makes the abstract categories of explicitation immediately understandable to the reader. The structure is logical, moving from motivation to formalization, methodology, results, and discussion.

---

### Weaknesses and Limitations

1.  **Data Source and Domain Limitations:** The paper is upfront about its limitations, but they are worth reiterating. The reliance on **TED-Multi and Europarl** means the corpus is biased towards formal, expository text. The types and frequencies of pragmatic explicitations in literary, conversational, or social media text could be drastically different. This limits the generalizability of the current findings and the corpus's applicability.

2.  **Scale of the Final Corpus:** While 2,700 annotated instances are a solid start, it is not a "large-scale" corpus by modern NLP standards. The sparsity of the phenomenon makes this understandable, but it may limit the training of data-hungry modern LLMs directly on this resource. Its primary value is likely as a gold-standard evaluation set or for fine-tuning in a few-shot setting.

3.  **Superficial Treatment of "Style":** The "Style" annotation (Replace vs. Add) is introduced but not deeply analyzed in the results. A more detailed discussion of how the *method* of integration (replacing vs. parenthetically adding) correlates with the *type* of explicitation or the source/target language pair would have been interesting and added another layer to the analysis.

4.  **Ambiguity in the Cross-lingual Test Sets:** A minor but notable weakness is the footnote explaining that the test sets for the cross-lingual evaluation (for PT, IT, FR, etc.) "only contain very clear cases of explicitations," whereas the German/Spanish test sets contain subtler cases. This makes the cross-lingual performance numbers not directly comparable to the monolingual ones and potentially overstates the model's performance on more challenging, subtle cases in new languages.

---

### Summary

**Novelty:** High. The paper tackles a clearly defined and under-researched problem with a sophisticated methodology.
**Significance:** High. The PragExTra corpus and the accompanying detection framework provide a foundational resource for research in culturally-aware machine translation and cross-lingual NLP.
**Clarity:** Excellent. The paper is well-structured, the concepts are well-explained with abundant examples, and the narrative is easy to follow.

In conclusion, this paper makes a substantial contribution by bridging a gap between translation theory and computational practice. Its main weaknesses are related to the inherent constraints of its data sources and scale, which the authors appropriately acknowledge and which provide clear directions for future work.

---

# Controlling Performance and Budget of a Centralized Multi-agent LLM System with Reinforcement Learning

Authors: Bowen Jin, TJ Collins, Donghan Yu, Mert Cemri, Shenao Zhang, Mengyu Li, Jay Tang, Tian Qin, Zhiyang Xu, Jiarui Lu, Guoli Yin, Jiawei Han, Zirui Wang

Keywords: Multi-agent LLM systems, Reinforcement Learning, Cost control, Budget-aware decision making, Centralized coordination, Performance-cost trade-off

Comments: 14 pages

Paper link: [http://arxiv.org/abs/2511.02755v1](http://arxiv.org/abs/2511.02755v1)

## Abstract

Large language models (LLMs) exhibit complementary strengths across domains and come with varying inference costs, motivating the design of multi-agent LLM systems where specialized models collaborate efficiently. Existing approaches predominantly rely on decentralized frameworks, which invoke multiple LLMs for every input and thus lead to substantial and uncontrolled inference costs. In this work, we introduce a centralized multi-LLM framework, where a controller LLM selectively coordinates a pool of expert models in a cost-efficient and cost-controllable manner. We formulate this coordination problem as reinforcement learning with dual objectives: maximizing task performance while minimizing the overall inference cost. In addition, we expect the multi-agent system to have adapted behavior with different budget conditions during inference. To this end, we propose CoRL, a reinforcement learning framework that optimizes the performance cost trade-off in a controllable multi-budget setting. Experiments on four diverse benchmarks demonstrate that CoRL enables a single system to surpass the best expert LLM under high-budget settings, while maintaining strong performance in more economical low-budget modes, highlighting the effectiveness of centralized coordination for scalable and cost-efficient multi-agent LLM systems.

## Summary

This paper introduces CoRL, a reinforcement learning framework for training cost-efficient and cost-controllable multi-agent LLM systems. The key innovation is a centralized architecture where a controller LLM selectively coordinates a pool of expert models, addressing the cost inefficiency of decentralized multi-LLM systems that typically invoke all experts for every input.

The method employs a centralized controller that decides whether to answer queries independently or decompose them into sub-queries for expert models. The framework is trained using PPO with dual reward signals: a performance reward based on task accuracy and a cost reward that penalizes exceeding predefined budget thresholds. To enable controllable behavior, the system incorporates budget-level conditioning in prompts during training, allowing the same model to operate under different budget modes (low, medium, high) at inference time.

Experimental results on math reasoning benchmarks demonstrate that CoRL achieves strong performance-cost trade-offs. In high-budget mode, the system surpasses the best individual expert model on three out of four datasets, while in low-budget mode, it maintains competitive performance while significantly reducing costs. The analysis reveals that the controller learns meaningful routing strategies, increasingly prioritizing higher-performance experts under larger budgets while avoiding expensive calls under constrained budgets. The learned behaviors generalize well to unseen data, confirming the effectiveness of the performance-cost reward design in shaping budget-aware decision-making.

## Critique

Of course. Here is a critique of the paper "Controlling Performance and Budget of a Centralized Multi-agent LLM System with Reinforcement Learning."

### Strengths

1.  **High Practical Relevance and Significance:** The paper addresses a critical, real-world problem: the escalating cost of running large-scale LLM systems. The focus on creating a single, adaptable system that can operate efficiently across different budget modes is highly significant for both research and industrial applications. The results demonstrate a clear and valuable performance-cost trade-off.

2.  **Novel Formulation and Clear Contribution:** The core idea—using RL to train a "controller" LLM to make cost-aware routing decisions within a multi-LLM system—is novel and well-formulated. The paper clearly distinguishes its "centralized" approach from existing "decentralized" multi-agent frameworks (where all agents are invoked for every input), framing its contribution around cost-efficiency and controllability.

3.  **Comprehensive and Convincing Experimental Setup:**
    *   The use of a diverse set of expert LLMs (o3, GPT-4.1, GPT-4.1-nano) with a clear performance-cost hierarchy provides a strong testbed.
    *   Evaluation across four different math reasoning benchmarks (MATH500, AMC, AIME) adds robustness to the claims.
    *   The comparison against strong baselines, including individual experts and a random routing baseline, effectively demonstrates the value of the learned routing policy.

4.  **Excellent Analysis and Insight:** The paper excels in its analysis section (Section 5). It doesn't just present final results but delves into the *dynamics* of the RL training process. The analysis of expert call ratios, reward components, and cost progression under different budget constraints provides deep insights into *how* and *why* the method works, which is invaluable for future research.

5.  **Clear and Well-Structured Presentation:** The paper is logically organized, with a clear introduction, well-defined framework, and a results section that builds from simple (2-LLM system) to complex (4-LLM system) setups. The figures are informative and directly support the textual arguments.

### Weaknesses

1.  **Limited Scope of Domains:** The evaluation is conducted exclusively on mathematical reasoning tasks. While this provides a controlled setting, it limits the generalizability of the claims. It remains to be seen if the learned cost-control strategies transfer effectively to other domains like creative writing, code generation, or complex dialogue, where the relationship between "expert" capability and cost might be less linear.

2.  **Training Cost and Complexity:** The paper acknowledges that the per-query cost *increases* during RL training as the controller learns to call experts more effectively. While the final system is cost-efficient at inference time, the upfront cost and complexity of the RL training process itself could be a significant barrier to adoption, a point that could be discussed more thoroughly.

3.  **Ablation and Sensitivity Analysis Gaps:**
    *   The performance of the system is highly dependent on the design of the system prompts (as shown in Figure 3). A more formal ablation study on prompt engineering would strengthen the paper, as the current results suggest the method might be sensitive to this initial setup.
    *   The choice of the cost budget `B` is crucial. While the paper explores different fixed values, it doesn't provide a clear methodology for how a practitioner should set this hyperparameter for a new task or set of experts.

4.  **Simplified Cost Model:** The cost model `c(y)` is based on a simple token-counting proxy using API prices. This is a reasonable simplification for a research paper, but real-world deployment costs can be more complex, involving latency, GPU memory, and other computational resources. The framework's effectiveness under a more nuanced cost function is an open question.

### Summary

This is a strong, well-executed paper that makes a meaningful contribution to the field of efficient LLM systems. Its primary strength lies in its novel formulation of the cost-control problem and its thorough experimental analysis, which provides compelling evidence for the approach's effectiveness. The main weaknesses are related to the scope of its evaluation and some practical considerations around training complexity and prompt sensitivity. Overall, it presents a promising direction for building scalable and economically viable multi-LLM applications.

