---
title: "ArXiv Daily Digest on 2025-10-09"
collection: digests
permalink: /digests/arxiv_cs_CL_2025-10-09_report
date: 2025-10-09
location: "Online"
---

Today's research highlights significant advancements in **multi-agent systems** and **multilingual AI**, revealing a clear trend toward collaborative intelligence and cross-lingual efficiency. A standout innovation is **Guided Topology Diffusion (GTD)**, which dynamically generates optimized communication structures for multiple LLM agents, balancing performance with cost efficiency. In multilingual domains, **Multilingual Generative Retrieval via Cross-lingual Semantic Compression (MGR-CSC)** introduces a novel framework that unifies semantically equivalent keywords across languages into "atoms," drastically reducing identifier space while improving retrieval accuracy. Meanwhile, **WaltzRL** refines safety alignment through multi-agent reinforcement learning, training a conversation agent and a feedback agent to collaboratively reduce unsafe outputs and overrefusals. These contributions underscore a broader movement toward more adaptive, resource-conscious, and robust AI systems.

## TL;DR

Total papers: 112 , Selected papers: 6

Here's a TL;DR summary of the key themes and insights from these papers:

**Main Themes:** Recent research focuses on enhancing AI systems through multi-agent collaboration, multilingual capabilities, and efficient knowledge sharing. Papers explore dynamic communication topologies, safety alignment through collaborative training, and cross-lingual knowledge compression.

**Key Insights:**
- **Multi-Agent Systems**: Papers propose frameworks for dynamic agent communication (https://arxiv.org/abs/2510.07799v1) and collaborative safety training (https://arxiv.org/abs/2510.08240v1), showing significant improvements in task performance and safety alignment through specialized agent interactions.

- **Multilingual Efficiency**: Research introduces methods for cross-lingual semantic compression in retrieval (https://arxiv.org/abs/2510.07812v1) and knowledge sharing in KG completion (https://arxiv.org/abs/2510.07736v1), achieving state-of-the-art results while reducing computational overhead.

- **Experience-Driven Learning**: The MUSE framework (https://arxiv.org/abs/2510.08002v1) demonstrates how agents can self-evolve through hierarchical memory mechanisms, enabling continuous improvement on long-horizon tasks.

- **Cost-Performance Tradeoffs**: Analysis reveals that while reasoning capabilities significantly improve negotiation performance (https://arxiv.org/abs/2510.08098v1), they come with substantial computational costs (400% increase for 31% performance gain).

**Common Trend**: There's a clear movement toward creating more adaptive, efficient, and collaborative AI systems that can learn from experience and optimize multi-objective tradeoffs between performance, cost, and safety.

*Note: All papers are from late 2024/early 2025 and represent cutting-edge research in their respective domains.*

---

# Multilingual Generative Retrieval via Cross-lingual Semantic Compression

Authors: Yuxin Huang, Simeng Wu, Ran Song, Yan Xiang, Yantuan Xian, Shengxiang Gao, Zhengtao Yu

Keywords: Multilingual Generative Retrieval, Cross-lingual Semantic Compression, Document Identifier (DocID), Information Retrieval, Semantic Atom Construction, Dynamic Constrained Decoding

Comments: EMNLP 2025, Findings, Long

Paper link: [http://arxiv.org/abs/2510.07812v1](http://arxiv.org/abs/2510.07812v1)

## Abstract

Generative Information Retrieval is an emerging retrieval paradigm that exhibits remarkable performance in monolingual scenarios.However, applying these methods to multilingual retrieval still encounters two primary challenges, cross-lingual identifier misalignment and identifier inflation. To address these limitations, we propose Multilingual Generative Retrieval via Cross-lingual Semantic Compression (MGR-CSC), a novel framework that unifies semantically equivalent multilingual keywords into shared atoms to align semantics and compresses the identifier space, and we propose a dynamic multi-step constrained decoding strategy during retrieval. MGR-CSC improves cross-lingual alignment by assigning consistent identifiers and enhances decoding efficiency by reducing redundancy. Experiments demonstrate that MGR-CSC achieves outstanding retrieval accuracy, improving by 6.83% on mMarco100k and 4.77% on mNQ320k, while reducing document identifiers length by 74.51% and 78.2%, respectively.

## Summary

Here is a summary of the paper "Multilingual Generative Retrieval via Cross-lingual Semantic Compression":

**Key Contributions:** This paper introduces MGR-CSC, a novel framework for multilingual generative information retrieval that addresses two main challenges: cross-lingual identifier misalignment and identifier inflation. The key innovations include a cross-lingual semantic compression approach that unifies semantically equivalent multilingual keywords into shared atoms, and a dynamic constrained multi-step decoding strategy that significantly reduces the decoding complexity during retrieval.

**Methods:** The MGR-CSC framework consists of three main components: (1) multilingual keyword extraction using LLMs to capture document semantics, (2) semantic atom construction through clustering of multilingual keywords in a shared semantic space, where semantically equivalent expressions across languages are assigned the same atom ID, and (3) dynamic constrained multi-step decoding that progressively generates document identifiers while narrowing the candidate space at each step. This approach transforms the retrieval process from searching over all documents to decoding within a compressed space of semantic atoms.

**Results:** Experiments on mMarco100k and mNQ320k datasets demonstrate that MGR-CSC achieves state-of-the-art performance, improving Recall@10 by 6.83% on mMarco100k and 4.77% on mNQ320k compared to existing methods. The framework also substantially reduces document identifier length by 74.51% and 78.2% on the respective datasets, indicating significant efficiency gains. The method shows consistent performance across diverse languages, including better handling of low-resource scenarios compared to baseline approaches like DSI-QG and SE-DSI.

## Critique

Of course. Here is a critique of the paper "Multilingual Generative Retrieval via Cross-lingual Semantic Compression," covering its strengths, weaknesses, and overall presentation.

### Overall Summary

This paper presents MGR-CSC, a novel framework for multilingual generative information retrieval (GIR) that addresses the key challenges of cross-lingual identifier misalignment and identifier inflation. The core idea—clustering semantically equivalent keywords across languages into shared "atoms" to create a compressed, language-agnostic document identifier (DocID) space—is both intuitive and powerful. The results are compelling, showing significant improvements in retrieval accuracy and substantial reductions in identifier space.

---

### Strengths

1.  **Novelty and Core Contribution:** The paper's primary strength is its clear and well-motivated solution to a specific, unaddressed problem in multilingual GIR. The concepts of "cross-lingual semantic compression" and the use of clustered "atoms" are novel in this context. Moving from language-specific keyword strings to a shared semantic ID space is a clever way to enforce cross-lingual alignment and combat the combinatorial explosion of DocIDs.

2.  **Significant and Comprehensive Results:** The empirical evaluation is thorough and convincing.
    *   **Performance:** The method achieves state-of-the-art or highly competitive results on two multilingual benchmarks (mMarco100k and mNQ320k), with impressive average gains in Recall@10. The fact that it performs consistently well across a diverse set of languages, including low-resource ones, is a strong point.
    *   **Efficiency:** The quantitative analysis of DocID usage (Figure 3) powerfully demonstrates one of the method's key benefits: a dramatic reduction (over 74%) in the DocID space. This directly addresses the "identifier inflation" problem and is a major practical advantage.
    *   **Ablation Studies:** The ablation study (Section 4.6) effectively isolates the contributions of the two main components (semantic compression and decoding strategy), showing that both are crucial to the method's success.

3.  **Clarity of Methodology:** The overall framework is explained clearly with the aid of a good conceptual diagram (Figure 2). The three-step process—keyword extraction, semantic atom construction, and dynamic constrained decoding—is logically presented and easy to follow. Algorithm 1 provides a concrete pseudocode implementation of the decoding process.

### Weaknesses and Areas for Improvement

1.  **Clarity of the Decoding Constraint:** While the high-level idea of dynamic constrained decoding is clear, the exact mechanism of `Constraint(K_i)` in Equation 5 and Algorithm 1 is somewhat vague. The paper states it "narrows the selection space based on constraints between atomic IDs from preceding steps," but a more precise explanation or a small example of how the candidate set `A_t` is pruned at each step would strengthen the technical description.

2.  **Comparison to a Stronger Baseline:** A notable omission is a comparison to a "vanilla" keyword-based GIR method that does **not** use semantic compression (i.e., using the raw, language-specific keywords as the DocID). While the ablation study shows that removing semantic compression hurts performance, a direct comparison to this as a baseline would more starkly illustrate the problem MGR-CSC solves (identifier misalignment/inflation) and the magnitude of its improvement.

3.  **Computational Cost of Preprocessing:** The method relies on several computationally heavy preprocessing steps: using an LLM (Llama3.1-8B) for keyword extraction and pseudo-query generation, and then running a clustering algorithm over all keywords in the corpus. The paper does not discuss the cost, scalability, or time required for this setup. For very large corpora, this could be a non-trivial overhead.

4.  **Dependence on Keyword Quality:** The entire framework's performance is predicated on the quality of the extracted keywords. If the LLM fails to extract salient, discriminative keywords for a document, the resulting DocID will be a poor representation. The paper assumes this step works well but does not analyze its potential failure modes or sensitivity.

### Assessment of Presentation

The paper is generally well-written and structured. The introduction effectively sets up the problem, and the related work section adequately contextualizes the research. The use of figures and tables is appropriate and helpful.

**Areas for improved clarity:**
*   As mentioned, the dynamic decoding constraint mechanism could be elaborated.
*   The description of the clustering process in Section 3.2 could be slightly more precise. The sentence "a fixed number of cluster centers is chosen" seems to conflict with the threshold-based approach described just before it. Clarifying whether the number of clusters `C` is fixed or determined by the threshold `θ` would be helpful.

### Conclusion

This is a strong paper that makes a valuable contribution to the field of multilingual generative retrieval. The proposed MGR-CSC method is novel, effectively solving two key challenges with an elegant solution of cross-lingual semantic compression. The results are significant, demonstrating superior performance and greatly improved efficiency over existing methods. While the description of the decoding process could be more precise and the computational cost of preprocessing acknowledged, these are minor issues in the context of a solid and impactful piece of research.

---

# The Price of Thought: A Multilingual Analysis of Reasoning, Performance, and Cost of Negotiation in Large Language Models

Authors: Sherzod Hakimov, Roland Bernard, Tim Leiber, Karl Osswald, Kristina Richert, Ruilin Yang, Raffaella Bernardi, David Schlangen

Keywords: Large Language Models, Multilingual Analysis, Negotiation, Reasoning, Strategic Adaptation, Computational Cost, Dialogue Games, Chain-of-Thought, Language Consistency

Comments: None

Paper link: [http://arxiv.org/abs/2510.08098v1](http://arxiv.org/abs/2510.08098v1)

## Abstract

Negotiation is a fundamental challenge for AI agents, as it requires an ability to reason strategically, model opponents, and balance cooperation with competition. We conduct the first comprehensive study systematically evaluating the effect of (LLM-)reasoning on the negotiation abilities of both commercial and open-weight LLMs, and do this across three languages. Using a self-play setup across three diverse dialogue games, we analyse trade-offs between performance and cost, the language consistency of reasoning processes, and the nature of strategic adaptation exhibited by models. Our findings show that enabling reasoning-that is, scaling test time compute-significantly improves negotiation outcomes by enhancing collaboration and helping models overcome task complexities, but comes at a substantial computational cost: reasoning improves GPT-5's performance by 31.4 % while increasing its cost by nearly 400 %. Most critically, we uncover a significant multilingual reasoning distinction: open-weight models consistently switch to English for their internal reasoning steps, even when negotiating in German or Italian (and thus possibly impacting potential explainability gains through the disclosure of reasoning traces), while leading commercial models maintain language consistency between their reasoning and final output.

## Summary

Of course. Here is a summary of the paper "The Price of Thought: A Multilingual Analysis of Reasoning, Performance, and Cost of Negotiation in Large Language Models."

**Key Contributions:** This paper presents the first comprehensive study to systematically evaluate the effect of Chain-of-Thought (CoT) reasoning on the negotiation abilities of both commercial and open-weight LLMs across multiple languages (English, German, and Italian). It specifically investigates the trade-offs between performance gains and computational costs, the language consistency of internal reasoning, and the nature of strategic adaptation in negotiation tasks.

**Methods:** The authors employ a self-play setup using the `clembench` framework, where two instances of the same LLM negotiate with each other. They use three distinct dialogue games requiring strategic reasoning, including "Deal or No Deal" (a bargaining scenario) and "Clean Up" (a cooperative object rearrangement task). The core experimental manipulation involves running models with their reasoning capabilities enabled (e.g., via CoT prompting) versus disabled, allowing for a direct comparison of performance, cost, and the language used in the reasoning traces.

**Key Results:**
1.  **Performance vs. Cost Trade-off:** Enabling reasoning significantly improves negotiation outcomes, helping models overcome task complexity and enhance collaboration. For instance, reasoning improved GPT-5's performance by 31.4%. However, this comes at a substantial computational cost, increasing GPT-5's cost by nearly 400%.
2.  **Critical Multilingual Distinction:** A major finding is a stark difference in language use during reasoning. Open-weight models consistently switch to English for their internal reasoning steps, even when negotiating in German or Italian. In contrast, leading commercial models maintain language consistency between their reasoning and final output. This has significant implications for the explainability and transparency of multilingual agents.
3.  **Strategic Adaptation:** The study analyzes whether models demonstrate genuine strategic adaptation or merely surface-level pattern matching, though the specific findings for this research question are not detailed in the provided excerpt.

In summary, this work highlights that while "thinking" (reasoning) is highly beneficial for complex tasks like negotiation, it is computationally expensive and its implementation reveals fundamental differences in how commercial and open-weight models handle multilingual contexts, potentially impacting their trustworthiness and applicability.

## Critique

Of course. Here is a critique of the strengths and weaknesses of the paper "The Price of Thought: A Multilingual Analysis of Reasoning, Performance, and Cost of Negotiation in Large Language Models".

### Strengths

1.  **High Novelty and Addressing Critical Gaps:** The paper's core contribution is highly novel. It is, as it claims, the first to systematically investigate the trade-off between reasoning (via Chain-of-Thought), performance, and computational cost in a negotiation context. Furthermore, extending this analysis to a multilingual setting (English, German, Italian) is a significant and underexplored area, moving beyond the typical English-centric evaluation.

2.  **Significant and Actionable Findings:** The results are substantial and have immediate implications for both research and practical deployment:
    *   **Quantifying the "Cost of Thought":** The finding that reasoning can improve GPT-5's performance by 31.4% at a 400% increase in cost provides a concrete, quantifiable trade-off that is crucial for real-world applications where inference budget is a constraint.
    *   **Revealing a Multilingual Reasoning Distinction:** The discovery that open-weight models default to English for internal reasoning, even when negotiating in other languages, is a critical finding. It highlights a potential "reasoning lingua franca" bias and raises important questions about the authenticity and explainability of multilingual reasoning traces in such models, contrasting them with commercial models that maintain language consistency.

3.  **Robust and Multi-faceted Evaluation Methodology:** The experimental design is a key strength. Using multiple, distinct dialogue games ("Deal or No Deal," "Clean Up") with different objectives (semi-competitive, cooperative) allows for a more comprehensive assessment of negotiation abilities beyond a single, narrow task. The self-play setup within a structured framework (clembench) ensures controlled and reproducible evaluation.

4.  **Clear and Well-Structured Presentation:** The paper is clearly written and logically organized. The research questions (RQ1-RQ3) are well-defined and directly guide the narrative. The inclusion of example game episodes (like Figure 1) effectively illustrates the experimental setup and interaction flow for the reader.

### Weaknesses

1.  **Limited Depth in Initial Sections:** While the introduction and related work sections are adequate, they could be more thorough. The related work section, in particular, cites many relevant papers but often summarizes them in a single sentence. A more detailed discussion of the specific methodologies and findings of the most closely related works would provide a stronger foundation and better contextualize the paper's novel contributions.

2.  **Incomplete Game Descriptions in Provided Text:** The provided text offers a good overview of "Deal or No Deal" but only a cursory mention of "Clean Up," noting that details are in an appendix not included here. For a reader assessing the paper based on this excerpt, the evaluation of the "Clean Up" game's suitability and mechanics remains unclear. A brief summary of its objectives and how it tests negotiation would strengthen the main text.

3.  **Ambiguity in "Strategic Adaptation" (RQ3):** The third research question—whether models demonstrate genuine strategic adaptation or surface-level pattern matching—is compelling but not yet substantiated by results in the provided text. The methodology for analyzing this (e.g., turn-by-turn reasoning analysis, probing for theory of mind) is not detailed, leaving it as a promising but currently unverified claim.

4.  **Scale of Evaluation (Potential Weakness):** The paper mentions 40 instances for the "Deal or No Deal" game (20 semi-competitive, 20 cooperative). Depending on the number of models and languages tested, this scale might be sufficient for clear trends, but it could be a limitation if the results are highly variable. A larger set of game instances would bolster the statistical significance of the findings.

### Summary

This paper presents a highly novel and timely investigation into a critical aspect of LLM agent deployment: the cost-benefit analysis of advanced reasoning capabilities in complex, interactive, and multilingual tasks. Its strengths lie in its unique research focus, robust experimental design, and significant, quantifiable findings—particularly the high cost of reasoning and the intriguing multilingual reasoning distinction. The main weaknesses are primarily in the depth of the background discussion and the incomplete presentation of the full experimental setup within the provided text. Overall, it addresses a clear gap in the literature and provides valuable insights for the field.

---

# Multilingual Knowledge Graph Completion via Efficient Multilingual Knowledge Sharing

Authors: Cunli Mao, Xiaofei Gao, Ran Song, Shizhu He, Shengxiang Gao, Kang Liu, Zhengtao Yu

Keywords: Multilingual Knowledge Graph Completion, Knowledge Sharing, Mixture-of-Experts, Iterative Entity Reranking, Cross-lingual Transfer

Comments: EMNLP 2025, Findings, Long Paper

Paper link: [http://arxiv.org/abs/2510.07736v1](http://arxiv.org/abs/2510.07736v1)

## Abstract

Large language models (LLMs) based Multilingual Knowledge Graph Completion (MKGC) aim to predict missing facts by leveraging LLMs' multilingual understanding capabilities, improving the completeness of multilingual knowledge graphs (KGs). However, existing MKGC research underutilizes the multilingual capabilities of LLMs and ignores the shareability of cross-lingual knowledge. In this paper, we propose a novel MKGC framework that leverages multilingual shared knowledge to significantly enhance performance through two components: Knowledge-level Grouped Mixture of Experts (KL-GMoE) and Iterative Entity Reranking (IER). KL-GMoE efficiently models shared knowledge, while IER significantly enhances its utilization. To evaluate our framework, we constructed a mKG dataset containing 5 languages and conducted comprehensive comparative experiments with existing state-of-the-art (SOTA) MKGC method. The experimental results demonstrate that our framework achieves improvements of 5.47%, 3.27%, and 1.01% in the Hits@1, Hits@3, and Hits@10 metrics, respectively, compared with SOTA MKGC method. Further experimental analysis revealed the properties of knowledge sharing in settings of unseen and unbalanced languages. We have released the dataset and code for our work on https://github.com/gaoxiaofei07/KL-GMoE.

## Summary

This paper introduces a novel framework for Multilingual Knowledge Graph Completion (MKGC) that addresses two key challenges: architectural mismatch between LLMs and knowledge-level tasks, and the discrepancy between text generation and entity ranking paradigms. The authors propose two main components: Knowledge-level Grouped Mixture of Experts (KL-GMoE) and Iterative Entity Reranking (IER).

The KL-GMoE architecture employs a grouped MoE design with knowledge-level expert routing to mitigate knowledge fragmentation while enhancing the model's capacity to capture cross-lingual shared knowledge. This design uses multiple expert groups where each group processes semantically similar information, with routing mechanisms that select specific experts based on input characteristics. The IER method modifies both training objectives and decoding strategies, enabling LLMs to iteratively refine entity rankings through multiple prediction rounds, thereby improving the utilization of multilingual shared knowledge.

The authors constructed a multilingual KG dataset spanning five languages (English, French, Italian, Japanese, Chinese) with over 3 million triples, reflecting natural knowledge distribution patterns including both shared and language-specific knowledge. Experimental results demonstrate significant improvements over state-of-the-art methods, achieving average gains of 5.47%, 3.27%, and 1.01% in Hits@1, Hits@3, and Hits@10 metrics respectively compared to the previous SOTA MKGC method. The framework also shows strong robustness to language imbalance and effective generalization to unseen languages, while maintaining computational efficiency with substantially fewer activated parameters compared to alternative approaches.

## Critique

### Summary of the Paper
The paper introduces a framework for Multilingual Knowledge Graph Completion (MKGC) that leverages Large Language Models (LLMs) to address two key challenges: (1) architectural mismatch between LLMs and MKGC tasks, and (2) the discrepancy between LLMs' text generation paradigm and the entity ranking requirements of MKGC. The proposed framework consists of two components: Knowledge-level Grouped Mixture of Experts (KL-GMoE), which models shared knowledge across languages, and Iterative Entity Reranking (IER), which enhances the utilization of this shared knowledge through iterative refinement. The authors construct a multilingual KG dataset and demonstrate state-of-the-art performance on MKGC metrics.

---

### Strengths

1. **Novelty of Approach**:
   - The KL-GMoE architecture is a novel adaptation of Mixture-of-Experts (MoE) for MKGC, featuring a knowledge-level routing mechanism to mitigate knowledge fragmentation and overload. This addresses a critical gap in applying LLMs to knowledge-intensive tasks.
   - The IER method introduces a fresh perspective on aligning LLMs with entity ranking tasks, enabling iterative refinement of candidate entities to improve ranking accuracy.

2. **Significance of Results**:
   - The framework achieves notable improvements over existing methods, with gains of 5.47% (Hits@1), 3.27% (Hits@3), and 1.01% (Hits@10) compared to the SOTA MKGC method. These results validate the effectiveness of the proposed components.
   - The framework demonstrates robustness in settings with imbalanced and unseen languages, highlighting its potential for real-world applications where language resources are unevenly distributed.

3. **Comprehensive Evaluation**:
   - The paper includes extensive experiments, including comparisons with embedding-based and generation-based baselines, architecture comparisons, parameter efficiency analysis, and ablation studies. This thorough evaluation strengthens the credibility of the claims.
   - The analysis of cross-lingual knowledge sharing and generalization to unseen languages provides valuable insights into the framework's capabilities.

4. **Clarity of Presentation**:
   - The paper is well-structured, with clear problem formulation, methodology descriptions, and visual aids (e.g., Figures 1–6) that enhance understanding.
   - The authors provide detailed implementation details, dataset statistics, and supplementary materials (e.g., prompt examples and algorithmic details in the appendix), ensuring reproducibility.

---

### Weaknesses

1. **Methodological Limitations**:
   - The framework relies on a candidate entity set generated by a Knowledge Graph Embedding (KGE) model (e.g., TransE). The performance bottleneck introduced by weaker KGE models (e.g., TransE vs. RotatE) is acknowledged but not fully resolved, limiting the framework's standalone capability.
   - The IER method introduces computational overhead during inference due to iterative decoding, which may hinder scalability for large-scale KGs.

2. **Scope and Generalizability**:
   - The framework is limited to textual KGs and cannot handle multimodal data, restricting its applicability to richer KG representations.
   - The reliance on LLMs' token limits prevents the framework from considering all entities in the KG during candidate selection, which may impact performance for large entity sets.

3. **Experimental Setup**:
   - The dataset, while multilingual, is limited to five languages (EN, FR, IT, JA, ZH). Expanding to more diverse or low-resource languages would further validate the framework's cross-lingual capabilities.
   - The use of LLaMA-2-7B as the base model may limit comparisons with larger or more advanced LLMs, which could exhibit different behaviors.

4. **Presentation Issues**:
   - Some sections, such as the routing mechanisms in KL-GMoE, are technically dense and may be challenging for readers unfamiliar with MoE architectures or LoRA-based fine-tuning.
   - The paper occasionally repeats key points (e.g., the improvements over SOTA), which could be streamlined for conciseness.

---

### Overall Assessment
The paper presents a well-motivated and innovative framework for MKGC, addressing critical challenges in leveraging LLMs for knowledge-intensive tasks. The proposed KL-GMoE and IER components are novel and empirically validated, with significant performance gains over existing methods. While the framework has limitations in scalability and generalizability, its contributions to multilingual knowledge sharing and robustness in imbalanced settings are noteworthy. The paper is clearly written and thoroughly evaluated, making it a valuable addition to the MKGC literature.

---

# Learning on the Job: An Experience-Driven Self-Evolving Agent for Long-Horizon Tasks

Authors: Cheng Yang, Xuemeng Yang, Licheng Wen, Daocheng Fu, Jianbiao Mei, Rong Wu, Pinlong Cai, Yufan Shen, Nianchen Deng, Botian Shi, Yu Qiao, Haifeng Li

Keywords: Self-evolving agents, Memory mechanisms, Long-horizon tasks, Experience-driven learning, LLM agents, Productivity automation

Comments: None

Paper link: [http://arxiv.org/abs/2510.08002v1](http://arxiv.org/abs/2510.08002v1)

## Abstract

Large Language Models have demonstrated remarkable capabilities across diverse domains, yet significant challenges persist when deploying them as AI agents for real-world long-horizon tasks. Existing LLM agents suffer from a critical limitation: they are test-time static and cannot learn from experience, lacking the ability to accumulate knowledge and continuously improve on the job. To address this challenge, we propose MUSE, a novel agent framework that introduces an experience-driven, self-evolving system centered around a hierarchical Memory Module. MUSE organizes diverse levels of experience and leverages them to plan and execute long-horizon tasks across multiple applications. After each sub-task execution, the agent autonomously reflects on its trajectory, converting the raw trajectory into structured experience and integrating it back into the Memory Module. This mechanism enables the agent to evolve beyond its static pretrained parameters, fostering continuous learning and self-evolution. We evaluate MUSE on the long-horizon productivity benchmark TAC. It achieves new SOTA performance by a significant margin using only a lightweight Gemini-2.5 Flash model. Sufficient Experiments demonstrate that as the agent autonomously accumulates experience, it exhibits increasingly superior task completion capabilities, as well as robust continuous learning and self-evolution capabilities. Moreover, the accumulated experience from MUSE exhibits strong generalization properties, enabling zero-shot improvement on new tasks. MUSE establishes a new paradigm for AI agents capable of real-world productivity task automation.

## Summary

Based on the provided paper "Learning on the Job: An Experience-Driven, Self-Evolving Agent for Long-Horizon Tasks", here is a summary focusing on its key contributions, methods, and results:

**Key Contributions:**
The paper introduces MUSE (Memory-Utilizing and Self-Evolving), a novel agent framework designed to address the limitation of existing LLM agents being "test-time static" - unable to learn from experience and continuously improve. The core contributions include: 1) An experience-driven closed-loop architecture that enables agents to evolve beyond their static pretrained parameters, 2) Autonomous conversion of raw action trajectories into structured, reusable memory without human intervention, and 3) Establishing new state-of-the-art performance on the challenging TAC benchmark for long-horizon productivity tasks.

**Methods:**
MUSE operates through a "Plan-Execute-Reflect-Memorize" iterative loop centered around a hierarchical Memory Module with three components: Strategic Memory (high-level behavioral paradigms), Procedural Memory (standard operating procedures for sub-tasks), and Tool Memory (individual tool usage guidance). The framework employs two specialized agents: a Planning-Execution Agent that decomposes tasks and executes actions using a minimal toolset, and a Reflect Agent that autonomously evaluates sub-task success and distills successful trajectories into structured memory. This design enables the agent to accumulate knowledge through interaction and reuse it for future tasks, with memory stored in natural language format for LLM-agnostic knowledge transfer.

**Results:**
The framework achieves remarkable results on the TAC benchmark, which features complex productivity tasks requiring over 40 action steps on average. MUSE establishes a new SOTA with a 51.78% partial completion score using only the lightweight Gemini-2.5 Flash model, representing a 20% relative improvement over previous methods. Experiments demonstrate strong continuous learning capabilities, with performance improving steadily over three iterations on repetitive tasks. The memory mechanism also shows excellent generalization, enabling zero-shot improvement on challenging unseen tasks by 10% compared to memory-less baselines. Ablation studies confirm the critical importance of both the reflection mechanism and the memory module to the framework's success.

## Critique

Of course. Here is a critical assessment of the paper "Learning on the Job: An Experience-Driven, Self-Evolving Agent for Long-Horizon Tasks."

### Strengths

1.  **High Novelty and Clear Problem Formulation:** The paper tackles a critical and timely limitation of current LLM agents: their static, "one-off" nature. The core idea of creating a "self-evolving" agent that learns from its own interaction history during deployment ("test-time learning") is highly novel and addresses a significant gap in making agents practical for real-world, repetitive tasks. The proposed "Plan-Execute-Reflect-Memorize" loop is a well-motivated and intuitive framework.

2.  **Comprehensive and Hierarchical Memory Design:** The Memory Module is the paper's standout contribution. Its hierarchical decomposition into **Strategic**, **Procedural**, and **Tool** memory is thoughtful. This structure effectively captures knowledge at different levels of abstraction, from high-level problem-solving strategies to low-level tool-usage patterns. The design choice to store memory in natural language, making it LLM-agnostic, is a significant strength for practicality and transferability.

3.  **Significant and Convincing Empirical Results:** The experimental results are a major strength. Achieving a new state-of-the-art (SOTA) on the challenging TAC benchmark by a large margin (51.78% vs. 43.19%) using a smaller, more cost-effective model (Gemini-2.5 Flash) is a powerful demonstration of the framework's efficacy. The continuous learning and generalization experiments provide strong, multi-faceted evidence:
    *   **Continuous Learning:** The monotonic performance improvement over three iterations on the `T_cl` subset provides clear, quantitative proof of the "self-evolving" capability.
    *   **Generalization:** The strong zero-shot performance improvement on the unseen `T_hard` set demonstrates that the agent is learning transferable skills and strategies, not just memorizing solutions.

4.  **Rigorous Experimental Design:** The paper includes well-designed ablation studies that convincingly validate the importance of key components (the Reflect Agent) and demonstrate the framework's adaptability to open-source models (DeepSeek-V3). This thoroughness strengthens the claims made.

### Weaknesses

1.  **Clarity and Presentation of the Core Mechanism:** While the overall framework is clear, the specific mechanisms for memory *creation* and *retrieval* could be explained more precisely.
    *   **Memory Creation:** The prompts or exact process used by the Reflect Agent to distill a raw trajectory into a structured SOP, Strategy, or Tool Memory entry are not detailed. The reader must trust that this process is robust and consistent.
    *   **Memory Retrieval:** The process for querying the Procedural Memory index is described at a high level, but the specific retrieval function (e.g., is it a simple keyword match, a semantic similarity search?) is not specified. This is a non-trivial engineering detail that impacts performance.

2.  **Limited Analysis of Failure Modes and Memory Limitations:** The paper rightly celebrates its successes but provides less insight into its limitations.
    *   What happens when the memory becomes large? Is there a risk of retrieving irrelevant or conflicting procedures? The paper mentions "deduplication and generalization" but does not elaborate on the algorithm or potential pitfalls.
    *   Could the agent learn and reinforce incorrect or sub-optimal procedures? The Reflect Agent seems to only validate against a predefined goal, but it's not clear how it handles "lucky" successes that are based on flawed reasoning.

3.  **Benchmark-Specific Evaluation:** The entire evaluation is based on the TAC benchmark. While TAC is an excellent choice for long-horizon tasks, it would strengthen the paper to show that the MUSE framework's benefits are not benchmark-specific. Demonstrating its effectiveness on another environment (e.g., WebArena, OSWorld) would significantly bolster the claim of general applicability.

4.  **Computational and Temporal Overhead:** The framework introduces significant complexity with two agents (PE and Reflect) and multiple reflection/memory-update steps. The paper does not discuss the computational cost or the time taken to complete tasks compared to a baseline agent. For real-world deployment, this overhead is a critical practical consideration.

### Overall Assessment

This is a strong and impactful paper. It identifies a fundamental problem with current LLM agents and proposes a novel, well-structured, and empirically validated solution. The demonstration of continuous learning and the achievement of a new SOTA are compelling. The main weaknesses lie in the opacity of some core algorithmic details and a lack of discussion on the framework's scaling limitations and computational cost. Despite these points, the significance of the results and the novelty of the approach make this a substantial contribution to the field of AI agents.

---

# Dynamic Generation of Multi-LLM Agents Communication Topologies with Graph Diffusion Models

Authors: Eric Hanchen Jiang, Guancheng Wan, Sophia Yin, Mengting Li, Yuchen Wu, Xiao Liang, Xinfeng Li, Yizhou Sun, Wei Wang, Kai-Wei Chang, Ying Nian Wu

Keywords: Multi-Agent Systems, Communication Topologies, Graph Diffusion Models, Dynamic Topology Generation, LLM Agents, Multi-Objective Optimization

Comments: None

Paper link: [http://arxiv.org/abs/2510.07799v1](http://arxiv.org/abs/2510.07799v1)

## Abstract

The efficiency of multi-agent systems driven by large language models (LLMs) largely hinges on their communication topology. However, designing an optimal topology is a non-trivial challenge, as it requires balancing competing objectives such as task performance, communication cost, and robustness. Existing frameworks often rely on static or hand-crafted topologies, which inherently fail to adapt to diverse task requirements, leading to either excessive token consumption for simple problems or performance bottlenecks for complex ones. To address this challenge, we introduce a novel generative framework called \textit{Guided Topology Diffusion (GTD)}. Inspired by conditional discrete graph diffusion models, GTD formulates topology synthesis as an iterative construction process. At each step, the generation is steered by a lightweight proxy model that predicts multi-objective rewards (e.g., accuracy, utility, cost), enabling real-time, gradient-free optimization towards task-adaptive topologies. This iterative, guided synthesis process distinguishes GTD from single-step generative frameworks, enabling it to better navigate complex design trade-offs. We validated GTD across multiple benchmarks, and experiments show that this framework can generate highly task-adaptive, sparse, and efficient communication topologies, significantly outperforming existing methods in LLM agent collaboration.

## Summary

Here is a concise summary of the paper "Dynamic Generation of Multi LLM Agents Communication Topologies with Graph Diffusion Models":

**Key Problem & Contribution:** The paper addresses the challenge of designing optimal communication topologies for multi-agent systems (MAS) powered by large language models (LLMs). Current systems typically use static or hand-crafted topologies (e.g., chain, star) that fail to adapt to varying task demands, leading to either excessive token consumption for simple tasks or performance bottlenecks for complex ones. The authors introduce **Guided Topology Diffusion (GTD)**, a novel framework that dynamically generates task-specific communication topologies using conditional graph diffusion models.

**Methodology:** GTD formulates topology generation as a conditional graph generation problem and consists of two core components:
1. A **surrogate reward model** (a Graph Neural Network) that predicts task utility and communication cost for a given topology without expensive simulation.
2. A **conditional graph diffusion generator** (a Graph Transformer) that learns to generate high-quality topologies through a denoising process.

The key innovation is **proxy-guided synthesis**: during inference, the diffusion process is steered at each step using zeroth-order optimization with the surrogate model. This allows real-time optimization toward multi-objective rewards (accuracy, cost, sparsity, robustness) without requiring differentiable objectives.

**Results & Evaluation:** Comprehensive experiments across multiple benchmarks (GSM8K, MATH, HumanEval, etc.) demonstrate that GTD:
- Achieves **state-of-the-art performance** on most tasks (e.g., 94.14% on GSM8K, 54.07% on MATH)
- Generates **highly cost-efficient** topologies, using significantly fewer tokens than baselines while maintaining high accuracy
- Shows **superior robustness** to agent failures, with minimal performance degradation compared to other methods
- Is **data-efficient**, achieving strong performance with limited training samples

The framework represents a significant advancement in adaptive multi-agent communication, moving beyond one-size-fits-all topologies to dynamically optimized structures that balance multiple competing objectives.

## Critique

Of course. Here is a commentary on the strengths and weaknesses of the paper "Dynamic Generation of Multi LLM Agents Communication Topologies with Graph Diffusion Models."

### **Strengths**

1.  **High Novelty and Problem Formulation:** The paper tackles a critical and underexplored problem in multi-agent systems: the automatic, dynamic design of communication topologies. Reframing this as a conditional graph generation problem is insightful. The core innovation lies in integrating a **proxy-guided, zeroth-order optimization** directly into the sampling process of a discrete graph diffusion model. This is a sophisticated approach that elegantly addresses the "black-box" and non-differentiable nature of the true reward function (which requires expensive simulation).

2.  **Significant and Comprehensive Results:** The empirical evaluation is thorough and compelling. The paper demonstrates state-of-the-art performance across multiple, diverse benchmarks (GSM8K, MATH, HumanEval, etc.), showing clear improvements over a wide array of strong baselines. More importantly, it doesn't just focus on accuracy; it provides strong evidence for the framework's **cost-efficiency** (significantly lower token consumption) and **robustness** (graceful degradation under agent failure). This multi-objective validation is crucial for proving the practical utility of the method.

3.  **Clear and Well-Structured Presentation:** The paper is exceptionally well-written and structured. The problem is motivated clearly, the methodology (GTD) is broken down into logical, digestible components (surrogate model, diffusion generator, proxy-guided synthesis), and the figures effectively illustrate the conceptual workflow and results. The inclusion of detailed ablation studies strengthens the paper by validating key design choices (e.g., the necessity of guidance, the choice of Graph Transformer).

### **Weaknesses**

1.  **Complexity and Computational Overhead:** While the results are impressive, the proposed framework is inherently complex and computationally heavy. It requires a multi-stage pipeline: dataset generation via simulation, training two separate models (a surrogate GNN and a graph diffusion model), and an inference process that involves running a multi-step diffusion process with a zeroth-order optimization loop (evaluating `K` candidates at each step). The paper does not provide a detailed analysis of the **end-to-end latency** for generating a topology compared to simpler baselines, which could be a practical concern for real-time applications.

2.  **Scalability to Larger Agent Teams:** The experiments are conducted with relatively small teams of agents (3-4). A key question is how well the method scales to systems with dozens or hundreds of agents. The complexity of the graph diffusion process and the surrogate model would grow with `N^2` (for the adjacency matrix), potentially making it intractable for very large `N`. An analysis or discussion of scalability limits would be valuable.

3.  **Dependence on Proxy Model Quality:** The entire guidance mechanism hinges on the accuracy of the lightweight surrogate model, `P_φ`. The paper theoretically bounds the performance gap based on the surrogate's error (in the appendix), but it doesn't empirically explore what happens when the surrogate's predictions are poor or when there is a significant domain shift between training and test tasks. The framework's robustness to a poorly calibrated proxy model could be a potential vulnerability.

4.  **Clarity on Baseline Comparison:** The paper compares against many methods, but it could be clearer about the **specific communication topology** used by each baseline. For instance, stating whether AgentVerse or G-Designer used a static topology or their own adaptive method in this specific experimental setup would help contextualize the comparisons more precisely.

### **Overall Assessment**

This is a **high-quality, impactful paper** that introduces a novel and powerful solution to a meaningful problem in multi-agent systems. The strengths significantly outweigh the weaknesses. The proposed GTD framework represents a substantial step beyond static or single-step generative approaches, offering a principled way to balance multiple competing objectives like performance, cost, and robustness. The results are convincing and comprehensively demonstrate the method's superiority. The weaknesses primarily point to interesting avenues for future work (scalability, efficiency optimizations) rather than fundamental flaws in the current contribution.

---

# The Alignment Waltz: Jointly Training Agents to Collaborate for Safety

Authors: Jingyu Zhang, Haozhu Wang, Eric Michael Smith, Sid Wang, Amr Sharaf, Mahesh Pasupuleti, Benjamin Van Durme, Daniel Khashabi, Jason Weston, Hongyuan Zhan

Keywords: Multi-agent reinforcement learning, Safety alignment, Collaborative agents, Dynamic Improvement Reward, LLM safety, Overrefusal reduction

Comments: None

Paper link: [http://arxiv.org/abs/2510.08240v1](http://arxiv.org/abs/2510.08240v1)

## Abstract

Harnessing the power of LLMs requires a delicate dance between being helpful and harmless. This creates a fundamental tension between two competing challenges: vulnerability to adversarial attacks that elicit unsafe content, and a tendency for overrefusal on benign but sensitive prompts. Current approaches often navigate this dance with safeguard models that completely reject any content that contains unsafe portions. This approach cuts the music entirely-it may exacerbate overrefusals and fails to provide nuanced guidance for queries it refuses. To teach models a more coordinated choreography, we propose WaltzRL, a novel multi-agent reinforcement learning framework that formulates safety alignment as a collaborative, positive-sum game. WaltzRL jointly trains a conversation agent and a feedback agent, where the latter is incentivized to provide useful suggestions that improve the safety and helpfulness of the conversation agent's responses. At the core of WaltzRL is a Dynamic Improvement Reward (DIR) that evolves over time based on how well the conversation agent incorporates the feedback. At inference time, unsafe or overrefusing responses from the conversation agent are improved rather than discarded. The feedback agent is deployed together with the conversation agent and only engages adaptively when needed, preserving helpfulness and low latency on safe queries. Our experiments, conducted across five diverse datasets, demonstrate that WaltzRL significantly reduces both unsafe responses (e.g., from 39.0% to 4.6% on WildJailbreak) and overrefusals (from 45.3% to 9.9% on OR-Bench) compared to various baselines. By enabling the conversation and feedback agents to co-evolve and adaptively apply feedback, WaltzRL enhances LLM safety without degrading general capabilities, thereby advancing the Pareto front between helpfulness and harmlessness.

## Summary

Of course. Here is a summary of the paper "The Alignment Waltz: Jointly Training Agents to Collaborate for Safety," focusing on its key contributions, methods, and results.

### Summary

This paper introduces **WaltzRL**, a novel multi-agent reinforcement learning (RL) framework designed to improve the safety alignment of large language models (LLMs) by addressing the fundamental tension between being helpful (avoiding overrefusal on benign prompts) and harmless (resisting adversarial attacks). The core idea is to formulate safety alignment as a collaborative, positive-sum game between two agents: a **conversation agent** that generates responses and a **feedback agent** that provides safety-focused feedback.

### Key Contributions

1.  **Multi-Agent RL Framework:** WaltzRL is a framework that jointly trains a conversation agent and a feedback agent to co-evolve, specializing in their respective roles.
2.  **Dynamic Improvement Reward (DIR):** A central innovation is the DIR for the feedback agent, which dynamically rewards it based on how much its feedback improves the conversation agent's subsequent response. This incentivizes the generation of useful, actionable feedback.
3.  **Adaptive Inference-Time Collaboration:** Unlike prior works that only deploy a single trained model, WaltzRL deploys both agents at inference. The feedback agent engages adaptively only when it deems the initial response unsafe or overrefusing, preserving low latency for safe queries.

### Methods

The WaltzRL method operates as follows:
- **Collaboration Protocol:** Given a user prompt, the conversation agent produces an initial response. The feedback agent then analyzes this response, outputs safety/overrefusal labels, and generates textual feedback if needed. The conversation agent can then revise its response based on this feedback.
- **Reward Shaping:** The conversation agent is rewarded for producing responses that are both safe and not overrefusing. The feedback agent's reward is a combination of a format reward, a label accuracy reward, and the crucial DIR.
- **Two-Stage Training:** The training process has two stages: first, the feedback agent is trained with a frozen conversation agent to learn correct formatting and labeling; second, both agents are trained collaboratively to refine their interaction and the usefulness of the feedback.

### Key Results

The proposed method was evaluated across several safety and overrefusal benchmarks and demonstrated state-of-the-art performance:

- **Enhanced Safety and Reduced Overrefusal:** WaltzRL significantly reduced both unsafe responses (Attack Success Rate dropped from 39.0% to 4.6% on WildJailbreak) and overrefusals (Over-Refuse Rate dropped from 45.3% to 9.9% on OR-Bench), outperforming all baselines including single-model RL and traditional safeguard models.
- **Preserved Helpfulness:** Crucially, WaltzRL achieved these safety gains with minimal degradation to general instruction-following and reasoning capabilities (e.g., on AlpacaEval and MMLU), despite not using helpfulness data during RL training.
- **Improved Efficiency:** The trained feedback agent triggers adaptively, leading to a low feedback rate (6.7%) on general prompts, making the system efficient for practical deployment.

In conclusion, WaltzRL advances the Pareto front between helpfulness and harmlessness by enabling two specialized agents to collaborate effectively, offering a promising path toward more nuanced and robust LLM safety alignment.

## Critique

Of course. Here is a critique of the paper "The Alignment Waltz: Jointly Training Agents to Collaborate for Safety."

### Overall Summary

This paper presents "WaltzRL," a novel multi-agent reinforcement learning framework designed to improve the safety of large language models (LLMs) by training a "conversation agent" and a "feedback agent" to collaborate. The core idea is to move beyond simple refusal-based safeguards and instead have the feedback agent provide constructive textual feedback, which the conversation agent learns to incorporate to revise its responses. The results are impressive, showing significant reductions in both unsafe responses and overrefusal across multiple benchmarks, with minimal degradation to general capabilities.

---

### Strengths

1.  **High Novelty in Formulation and Reward Design:**
    *   **Positive-Sum Collaboration:** The paper's greatest strength is its reformulation of safety alignment as a collaborative, positive-sum game between two agents. This is a significant departure from the more common adversarial or zero-sum training paradigms (e.g., attacker vs. defender). The agents are incentivized to work together to achieve a common goal.
    *   **Dynamic Improvement Reward (DIR):** The DIR is a clever and central contribution. By rewarding the feedback agent based on the *improvement* it induces in the conversation agent's subsequent response, the framework directly incentivizes useful, actionable feedback. This creates a dynamic where the two agents co-adapt and learn from each other's evolving policies.

2.  **Significant and Comprehensive Empirical Results:**
    *   The paper demonstrates a clear advancement of the Pareto frontier between helpfulness and harmlessness. The reported reductions in Attack Success Rate (e.g., 39.0% → 4.6% on WildJailbreak) and Over-Refuse Rate (e.g., 45.3% → 9.9% on OR-Bench) are substantial.
    *   The evaluation is thorough, using five diverse datasets to measure safety, overrefusal, and general capabilities (AlpacaEval, MMLU, etc.). This provides strong evidence for the method's effectiveness and its lack of detrimental side effects on general performance.
    *   The inclusion of a well-chosen set of baselines, including an "Oracle" baseline, strengthens the claims. The finding that the oracle underperforms WaltzRL is a powerful argument for the necessity of detailed, learned feedback over simple label-based instructions.

3.  **Practical Deployment Considerations:**
    *   The paper thoughtfully addresses the practical issue of latency through the "Feedback Trigger Rate" (FTR). The two-stage training and the inclusion of classification labels (`unsafe`, `overrefuse`) allow the system to be adaptive, only invoking the feedback loop when necessary. The low FTR (6.7%) on general prompts makes the approach more feasible for real-world applications.

4.  **Clear and Well-Structured Presentation:**
    *   The paper is generally well-written. The high-level analogy of a "waltz" is effective. The collaboration protocol (Section 2.1) and the multi-agent RL algorithm (Section 2.3) are described with sufficient detail, and Figure 1 provides a helpful overview.

---

### Weaknesses and Areas for Improvement

1.  **Complexity and Computational Cost:**
    *   The primary weakness of WaltzRL is its inherent complexity and computational overhead. Training two agents simultaneously with multi-turn rollouts is significantly more expensive than single-agent RL or supervised fine-tuning. While the adaptive inference mitigates *inference-time* cost, the *training cost* is a major barrier to entry for many research groups.
    *   The paper would be strengthened by a more explicit discussion of the computational resources required for training (e.g., GPU hours).

2.  **Ablations and Analysis Could Be Deeper:**
    *   While the ablation on the reward design (Section 3.3) is valuable, more could be done. For instance:
        *   **Round Limit:** The choice of `T_max = 1` is pragmatic but somewhat arbitrary. An ablation studying the effect of allowing more feedback rounds during training or inference would be interesting.
        *   **Agent Specialization:** It would be insightful to analyze the emergent specialization of the two agents. Does the feedback agent develop a distinct "safety reasoning" capability compared to the base model?
    *   The analysis of the "emergent behavior" where the feedback agent generates an "ideal response" is intriguing but brief. A deeper qualitative analysis of these emergent collaboration patterns would be fascinating.

3.  **Clarity on the "Judge" and Potential Circularity:**
    *   The "Alignment Labels" from the LLM judge (used to compute `R_c` and `R_f_label`) are critical to the entire training process. The paper mentions they are derived from an LLM judge but provides details only in the appendix. The reliability and potential biases of this judge are a potential point of failure. There is a risk of circularity if the judge's limitations are baked into the trained agents.

4.  **Limited Discussion of Failure Modes:**
    *   The paper focuses on successes. A more detailed discussion of the limitations and failure cases (beyond the short appendix section) would provide a more balanced view. Under what conditions does the collaboration break down? Are there types of prompts where the feedback agent provides unhelpful or even harmful guidance?

### Conclusion

This is a highly compelling and novel paper that makes a significant contribution to the field of AI safety. The core idea of collaborative, positive-sum multi-agent RL for alignment is powerful and well-executed. The empirical results are strong and comprehensively demonstrate a superior trade-off between safety and helpfulness. The main weaknesses lie in the method's complexity and cost, and a somewhat surface-level treatment of its limitations and failure modes. Despite this, "WaltzRL" represents a clear step forward and will likely inspire considerable follow-up research.

