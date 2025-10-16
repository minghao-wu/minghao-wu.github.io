---
title: "ArXiv Daily Digest on 2025-10-15"
collection: digests
permalink: /digests/arxiv_cs_CL_2025-10-15_report
date: 2025-10-15
location: "Online"
---

Today's research landscape reveals a strong focus on enhancing the efficiency and reliability of large language models (LLMs), with several papers introducing novel optimization frameworks. A key trend is the move beyond simple correctness metrics towards more nuanced objectives like faithful reasoning in retrieval-augmented generation and robust preference optimization for machine translation. We see significant progress in parameter-efficient methods, including a new technique for sparse subnetwork enhancement to boost performance in underrepresented languages and a parameter-free approach called GatePro for improving expert selection diversity in Mixture-of-Experts (MoE) models. Concurrently, safety remains a priority, evidenced by DSCD (Detoxification with Self-Constrained Decoding), a lightweight decoding-time method for model detoxification. These works collectively highlight the field's push towards more specialized, trustworthy, and resource-conscious model development.

## TL;DR

Total papers: 69 , Selected papers: 5

Here's a TL;DR summary of the key themes and insights from these papers:

**Main Themes:**
- **Preference Optimization & Model Alignment**: Papers explore advanced preference optimization techniques for machine translation (M^2PO) and retrieval-augmented generation (VERITAS), moving beyond simple win-loss pairs to multi-perspective rewards and process supervision.
- **Efficient Model Adaptation**: Several papers focus on parameter-efficient methods, including sparse subnetwork fine-tuning for low-resource languages and parameter-free expert selection in MoE models.
- **Safety & Reliability**: Work on detoxification (DSCD) and faithful reasoning emphasizes building more trustworthy and transparent LLMs.

**Key Insights:**
- **M^2PO** (https://arxiv.org/abs/2510.13434) introduces multi-perspective rewards combining hallucination penalties with dynamic quality scoring, achieving SOTA in MT while addressing QE model limitations.

- **GatePro** (https://arxiv.org/abs/2510.13079) provides parameter-free expert diversity optimization for MoE models, preventing redundant expert activation through localized competition mechanisms.

- **DSCD** (https://arxiv.org/abs/2510.13183) offers plug-and-play detoxification by manipulating internal layer distributions during decoding, maintaining fluency while reducing toxicity.

- **VERITAS** (https://arxiv.org/abs/2510.13272) introduces process-based rewards for faithful reasoning in RAG systems, significantly improving reasoning transparency without sacrificing performance.

- **Sparse Subnetwork** (https://arxiv.org/abs/2510.13580) enhances low-resource language performance by fine-tuning only language-specific neurons (0.2-1% parameters), preserving general capabilities while improving cross-lingual alignment.

These papers collectively advance efficient, safe, and multilingual LLM deployment through innovative optimization strategies and targeted architectural interventions.

---

# Beyond Single-Reward: Multi-Pair, Multi-Perspective Preference Optimization for Machine Translation

Authors: Hao Wang, Linlong Xu, Heng Liu, Yangyang Liu, Xiaohu Zhao, Bo Zeng, Liangying Shao, Longyue Wang, Weihua Luo, Kaifu Zhang

Keywords: Preference Optimization, Machine Translation, Multi-Perspective Reward, Multi-Pair Construction, Hallucination Mitigation, Direct Preference Optimization

Comments: None

Paper link: [http://arxiv.org/abs/2510.13434v1](http://arxiv.org/abs/2510.13434v1)

## Abstract

Direct Preference Optimization (DPO) is a powerful paradigm for aligning Large Language Models (LLMs) to human preferences in Machine Translation (MT), but current methods are hindered by two fundamental challenges: (1) flawed reward signals from Quality Estimation (QE) models that overlook critical errors like translation hallucination, and (2) inefficient data utilization that discards valuable learning signals by selecting only a single win-loss pair. To address these limitations, we introduce M^2PO: Multi-Pair, Multi-Perspective Preference Optimization. Our framework integrates a multi-perspective reward engine that creates a more robust signal by combining two key viewpoints: a new hallucination penalty for factuality, and an innovative dynamic quality score that adaptively fuses external evaluations with the model's own evolving judgment. This is synergistically paired with a multi-pair construction strategy that systematically creates a comprehensive set of preference pairs from the entire pool of translation candidates. This synergistic approach ensures the model learns from a richer spectrum of quality trade-offs, leading to more robust and faithful translations. On challenging WMT21-22 benchmarks, M^2PO substantially outperforms existing preference optimization methods and demonstrates highly competitive performance against leading proprietary LLMs.

## Summary

This paper introduces **M2PO (Multi-Pair, Multi-Perspective Preference Optimization)**, a novel framework designed to address two fundamental limitations in current preference optimization methods for machine translation: (1) unreliable reward signals from Quality Estimation (QE) models that often overlook critical errors like translation hallucinations, and (2) inefficient data utilization where methods typically select only a single win-loss pair, discarding valuable learning signals.

The key contributions are threefold. First, the authors identify and analyze the weaknesses of QE-based rewards, demonstrating through empirical analysis that standard QE metrics struggle with partial hallucinations and exhibit scoring instability for nuanced translation errors. Second, they propose M2PO's integrated innovations: a **multi-perspective reward engine** that combines a dedicated hallucination penalty with an innovative dynamic scoring curriculum, and a **multi-pair optimization strategy** that systematically creates comprehensive preference pairs from the entire candidate pool. Third, the framework employs a **multi-component optimization objective** consisting of dynamic multi-pair DPO loss, listwise ranking loss, and behavior cloning loss to ensure robust learning while maintaining generative quality.

Experiments on WMT21-22 benchmarks show that M2PO substantially outperforms existing preference optimization methods (including standard DPO and CPO) and demonstrates highly competitive performance against leading proprietary LLMs like GPT-4o. The 7B model enhanced with M2PO even surpasses GPT-4o-mini (one of its data sources) while maintaining comparable faithfulness, achieving a "student-surpasses-teacher" outcome. Additional analyses confirm M2PO's generalization across various DPO-like algorithms and the importance of its core components through comprehensive ablation studies.

## Critique

Of course. Here is a commentary on the strengths and weaknesses of the paper "Beyond Single-Reward: Multi-Pair, Multi-Perspective Preference Optimization for Machine Translation."

### **Strengths**

1.  **Clear Problem Identification and Motivation:** The paper excels at diagnosing two specific, well-known pain points in applying preference optimization to Machine Translation (MT):
    *   The unreliability of Quality Estimation (QE) models, particularly their blindness to nuanced errors like partial hallucinations.
    *   The data inefficiency of using only a single (win, lose) pair from a larger candidate pool.
    The analysis in Section 3, including the correlation study on the HalOmi benchmark and the volatility plot, provides strong, empirical motivation for the work.

2.  **Novel and Well-Integrated Framework (M2PO):** The proposed M2PO framework is not just an incremental improvement but a holistic system that elegantly addresses the identified problems.
    *   **Multi-Perspective Reward:** The combination of a standard QE score with a dedicated, word-alignment-based "factuality bonus" is a direct and clever solution to the hallucination problem. The "Dynamic Preference Modeling" that fuses this static score with the model's own evolving judgment is a sophisticated curriculum learning approach that prevents reward hacking.
    *   **Multi-Pair Construction:** The "head-to-tail" pairing strategy is a simple yet effective way to maximize the utility of the candidate set, ensuring the model learns from a wider spectrum of quality differences.

3.  **Comprehensive and Significant Results:** The experimental section is thorough and convincing.
    *   The model achieves state-of-the-art results among open-source models and demonstrates highly competitive performance against top-tier proprietary models like GPT-4o, which is a significant accomplishment for a 7B parameter model.
    *   The evaluation is multi-faceted, using both reference-based (COMET-22) and reference-free (XCOMET) metrics, and crucially, a separate "Coverage Score" to explicitly measure faithfulness, directly addressing the paper's core motivation.
    *   The "student-surpasses-teacher" result (beating GPT-4o-mini, a source of its training data) is a powerful testament to the effectiveness of the distillation and optimization process.

4.  **Excellent Analysis and Ablations:** The paper goes beyond main results to provide deep insights.
    *   The demonstration that M2PO's benefits generalize across different DPO-like algorithms (DPO, KTO, SimPO, etc.) positions it as a versatile, algorithm-agnostic *framework* rather than just a new algorithm, significantly increasing its impact.
    *   The ablation studies are critical and clearly show the contribution of each component, with the catastrophic drop from removing the Behavior Cloning loss (`L_BC`) being a particularly stark illustration of the challenges of preference optimization.

### **Weaknesses**

1.  **Complexity and Computational Cost:** The framework introduces significant complexity and computational overhead, which is not deeply discussed as a limitation. The pipeline requires:
    *   Generating a large candidate set from multiple models (including API calls to proprietary models).
    *   Running two offline scoring models per candidate (a QE model and a word-aligner for the factuality bonus).
    *   Performing dynamic score calculation and multi-pair construction during training.
    While the results justify the cost, a discussion of the trade-offs and the feasibility for researchers with limited resources would have been valuable.

2.  **Hyperparameter Sensitivity:** The method relies on a non-trivial number of new hyperparameters (`λ_f`, `α_t` scheduler, `β`, `τ_s`, `τ_w`, and the three loss weights `λ_pref`, `λ_rank`, `λ_bc`). The paper states the values used but does not discuss how sensitive the performance is to these choices or the effort required to tune them. This could be a barrier to adoption.

3.  **Clarity and Presentation:**
    *   **Figure 2:** The central workflow figure is informative but very dense. The connection between the "Dynamic Multi-Perspective Evaluator" and the "Static Score (r_s)" could be clearer, as the figure might lead a reader to think the dynamic fusion happens during the offline scoring stage, not during training.
    *   **Repetitive Text:** The paper has some instances of repetitive phrasing, particularly in the introduction and conclusion, where the contributions and results are restated multiple times in very similar language. A more concise presentation would improve readability.

4.  **Novelty of Components:** While the *integration* is novel and powerful, the individual components are built on established ideas: multi-pair learning resembles techniques from learning-to-rank, dynamic weighting is a form of curriculum learning, and the composite loss is a common regularization strategy. The paper could more explicitly delineate its conceptual innovations from the prior art it builds upon.

### **Overall Assessment**

This is a strong, high-impact paper that makes a significant contribution to the field of Machine Translation and preference learning. It identifies critical, unsolved problems and proposes a sophisticated, well-motivated, and empirically validated framework that delivers state-of-the-art results. The main weaknesses relate to the practical complexities of the method and minor issues in presentation, but they do not detract from the core significance of the work. M2PO represents a substantial step forward in building more reliable and data-efficient preference-optimized translation models.

---

# GatePro: Parameter-Free Expert Selection Optimization for Mixture-of-Experts Models

Authors: Chen Zheng, Yuhang Cai, Deyi Liu, Jin Ma, Yiyuan Ma, Yuan Yang, Jing Liu, Yutao Zeng, Xun Zhou, Siyuan Qiao

Keywords: Mixture-of-Experts, Expert Selection Diversity, Parameter-Free Optimization, Sparse Activation, Gating Mechanism, Competitive Propagation, Load Balancing, Expert Utilization

Comments: None

Paper link: [http://arxiv.org/abs/2510.13079v1](http://arxiv.org/abs/2510.13079v1)

## Abstract

Modern large language models leverage Mixture-of-Experts (MoE) architectures for efficient scaling, but face a critical challenge: functionally similar experts are often selected simultaneously, creating redundant computation and limiting effective model capacity. Existing auxiliary balance loss methods improve token distribution but fail to address the underlying expert diversity problem. We introduce GatePro, a novel parameter-free method that directly promotes expert selection diversity. GatePro identifies the most similar expert pairs and introduces localized competition mechanisms, preventing redundant expert co-activation while maintaining natural expert specialization. Our comprehensive evaluation demonstrates GatePro's effectiveness across model scales and benchmarks. Analysis demonstrates GatePro's ability to achieve enhanced expert diversity, where experts develop more distinct and complementary capabilities, avoiding functional redundancy. This approach can be deployed hot-swappable during any training phase without additional learnable parameters, offering a practical solution for improving MoE effectiveness.

## Summary

Based on the provided paper, here is a summary of "GatePro: Parameter-Free Expert Selection Optimization for Mixture-of-Experts Models."

**Key Problem & Contribution:** Modern large language models (LLMs) increasingly use Mixture-of-Experts (MoE) architectures for efficient scaling. A critical, overlooked challenge in these models is the lack of **expert selection diversity**, where functionally similar experts are often activated simultaneously. This leads to redundant computation and limits the model's effective capacity. While existing methods use auxiliary balance losses to ensure tokens are distributed evenly across experts, they fail to address this underlying issue of functional redundancy. The key contribution of this paper is **GatePro**, a novel, **parameter-free** method that directly promotes expert selection diversity by preventing similar experts from being co-activated.

**Method:** GatePro operates by introducing a localized competition mechanism. Its process is as follows:
1.  **Gate Similarity Computation:** It first computes a cosine similarity matrix based on the gating weight vectors to identify the most similar pairs of experts.
2.  **Localized Competition:** For each input token, and for every pair of highly similar experts, GatePro allows them to "compete." It compares their gating logits for the current token and applies a constant negative penalty to the logit of the "losing" expert. This effectively suppresses it from being selected, forcing the routing mechanism to choose more diverse, functionally distinct experts.
A major advantage is that GatePro is **hot-swappable**—it can be enabled or disabled during any phase of training without introducing new learnable parameters or requiring retuning.

**Key Results:** The authors conduct extensive experiments on various model scales (e.g., 0.7B/7B and 1.3B/13B parameters) and benchmarks (MMLU-Pro, GSM8K, BBH, etc.).
*   **Performance:** GatePro consistently outperforms baseline MoE models across all stages of training (from early pre-training to continuous training) and on nearly all evaluated tasks. The improvements are particularly notable in reasoning-intensive tasks like arithmetic (GSM8K) and code generation (MBPP).
*   **Generalizability:** The method's effectiveness is also demonstrated on the open-source OLMoE architecture, confirming it is not limited to a single model implementation.
*   **Mechanistic Analysis:** Analysis shows that GatePro accelerates expert activation during training, reduces the cosine similarity between expert gating weights, and increases selection entropy. This confirms that experts under GatePro develop more distinct and complementary specializations, especially in deeper layers of the network where specialization is most critical.

## Critique

Of course. Here is a critique of the paper "GatePro: Parameter-Free Expert Selection Optimization for Mixture-of-Experts Models".

### Strengths

1.  **Novelty and Problem Identification:** The paper's core strength is its clear identification of a problem often overlooked in MoE research: **functional redundancy among co-activated experts**. While previous work focused on load balancing (ensuring experts get a similar number of tokens), this paper correctly argues that this does not guarantee the *selected set* of experts for a given token is diverse. The concept of promoting "expert selection diversity" is a fresh and valuable perspective.

2.  **Elegant and Parameter-Free Solution:** The proposed method, GatePro, is conceptually elegant and technically simple. By computing pairwise expert similarities from the gating weights and applying a localized competition penalty, it directly tackles the identified problem. Crucially, its "parameter-free" and "hot-swappable" nature is a significant practical advantage, as it can be inserted into existing training pipelines without adding learnable parameters or destabilizing training, making adoption very easy.

3.  **Comprehensive and Convincing Evaluation:** The experimental section is thorough and well-designed. The authors evaluate across:
    *   **Multiple model scales** (0.7B/7B and 1.3B/13B).
    *   **Multiple training stages** (from early pre-training to continued training).
    *   **Multiple benchmarks** covering reasoning, knowledge, and coding.
    *   **Multiple architectures** (their internal Seed-MoE and the open-source OLMoE).
    This multi-faceted approach strongly supports the claim that GatePro is a robust and generally applicable method.

4.  **Strong and Consistent Results:** The performance improvements, while not massive, are **consistent and statistically significant** across almost all tasks and scales. The gains in arithmetic (GSM8K) and coding (MBPP) are particularly notable, suggesting that tasks requiring precise computation benefit more from diverse expert selection. The fact that improvements are shown from the very early stages of training is a powerful argument for its efficacy.

5.  **Excellent Mechanistic Analysis:** The paper goes beyond mere performance metrics to provide a deep dive into *how* GatePro works. The analysis of zero-token counts, cosine similarity, angular separation, and spectral entropy provides compelling evidence that the method successfully accelerates expert utilization and enforces functional diversity, especially in the critical deeper layers of the network.

### Weaknesses

1.  **Limited Ablation and Hyperparameter Sensitivity:** The paper lacks a thorough ablation study. A key hyperparameter is the penalty value `λ`, which is set to `10^-4` with little justification. It is unclear how sensitive the method is to this value. Furthermore, while the localized competition is applied to the *most similar* pair, the impact of competing with the top-*k* most similar experts is not explored. Understanding these design choices is important for future work.

2.  **Computational Overhead Omitted:** The authors correctly note that the cosine similarity computation has `O(N²d)` complexity but then dismiss the overhead as "minimal." For models with a large number of experts `N` (e.g., 256 or more) and high dimensionality `d`, this quadratic cost could become non-trivial, especially if the similarity matrix is recomputed frequently. A more honest discussion of this cost and potential optimizations (e.g., how often `S` is recalculated) would be beneficial.

3.  **Clarity of Presentation (Minor):** While the overall presentation is clear, the writing can be slightly repetitive in places, particularly in the results section where the same pattern of improvement is described for multiple benchmarks. The paper could be more concise. Additionally, the term "competitive propagation" is introduced but not fully fleshed out; it essentially describes the chain reaction of one expert's suppression affecting the selection of others, but this could be explained more intuitively.

4.  **Theoretical Grounding:** The method is largely empirical and heuristic. While the results are convincing, a more rigorous theoretical justification for why penalizing the most similar pair is the optimal strategy, or how this relates to concepts like the capacity of the expert pool, would strengthen the paper's foundation.

### Overall Assessment

This is a **strong, high-quality paper** that makes a meaningful contribution to the field of MoE models. It identifies a genuine limitation in current approaches and provides a simple, effective, and practical solution. The novelty of focusing on "selection diversity" rather than just "load balance" is significant. The comprehensive evaluation and insightful mechanistic analysis make the results highly convincing. While it has minor weaknesses regarding ablations and computational discussion, its strengths far outweigh them. This paper is likely to influence future MoE research and could see rapid adoption in practice due to its parameter-free, hot-swappable design.

---

# DSCD: Large Language Model Detoxification with Self-Constrained Decoding

Authors: Ming Dong, Jinkui Zhang, Bolong Zheng, Xinhui Tu, Po Hu, Tingting He

Keywords: Large Language Model Detoxification, Self-Constrained Decoding, Toxic Layer Localization, Safe Text Generation, Inference-time Safety Enhancement

Comments: Accepted at EMNLP 2025 MainConference

Paper link: [http://arxiv.org/abs/2510.13183v1](http://arxiv.org/abs/2510.13183v1)

## Abstract

Detoxification in large language models (LLMs) remains a significant research challenge. Existing decoding detoxification methods are all based on external constraints, which require additional resource overhead and lose generation fluency. This work proposes Detoxification with Self-Constrained Decoding (DSCD), a novel method for LLM detoxification without parameter fine-tuning. DSCD strengthens the inner next-token distribution of the safety layer while weakening that of hallucination and toxic layers during output generation. This effectively diminishes toxicity and enhances output safety. DSCD offers lightweight, high compatibility, and plug-and-play capabilities, readily integrating with existing detoxification methods for further performance improvement. Extensive experiments on representative open-source LLMs and public datasets validate DSCD's effectiveness, demonstrating state-of-the-art (SOTA) performance in both detoxification and generation fluency, with superior efficiency compared to existing methods. These results highlight DSCD's potential as a practical and scalable solution for safer LLM deployments.

## Summary

Here is a summary of the paper "DSCD: Large Language Model Detoxification with Self-Constrained Decoding":

**Key Contributions:**
The paper introduces DSCD (Detoxification with Self-Constrained Decoding), a novel method for detoxifying large language models (LLMs) without parameter fine-tuning. DSCD's key contributions include: 1) A lightweight, plug-and-play approach that maintains high compatibility with existing methods; 2) Two operational modes - MODE-1 for precise toxic region localization and high performance, and MODE-2 for efficiency with static toxic layers; 3) State-of-the-art results in both detoxification effectiveness and generation fluency.

**Methods:**
DSCD operates by adjusting the next-token distribution during decoding, strengthening safety layers while weakening toxic and hallucination layers. The method leverages early exit capabilities to access intermediate layer distributions and uses Jensen-Shannon divergence to identify toxic regions at the token level (unlike previous sequence-level approaches). MODE-1 dynamically locates toxic, safety, and hallucination layers for each token, while MODE-2 uses pre-identified static toxic layers for efficiency. The approach applies self-constraints by subtracting toxic region distributions from factual region distributions in the log domain.

**Results:**
Extensive experiments on multiple datasets (SafeEdit, HarmfulQA, Advbench, etc.) and models (Llama2, Qwen2, Mistral) show that DSCD alone improves detoxification performance by an average of 11.78% over vanilla models. When integrated with existing methods like DINM and SafeDecoding, it provides additional improvements of 3.70-4.03%. Crucially, DSCD maintains or even improves generation fluency compared to baseline methods and achieves these results with minimal computational overhead - MODE-2 runtime is close to vanilla models and significantly faster than methods like DINM. The approach also shows consistent performance gains across different evaluation metrics and classifiers (RoBERTa and GPT-4o).

## Critique

Of course. Here is a critique of the paper "DSCD: Large Language Model Detoxification with Self-Constrained Decoding," focusing on its strengths and weaknesses.

### **Strengths**

1.  **Novelty and Core Idea:** The paper's central premise is highly novel and compelling. The idea of performing "self-constrained decoding" by leveraging the model's own internal layers (toxic, safety, hallucination) for detoxification, without relying on external models or fine-tuning, is a significant departure from existing methods like SafeDecoding or DINM. This internal, "self-referential" approach is a clever and elegant contribution.

2.  **Practicality and Efficiency:** The paper effectively addresses key limitations of prior work. By eliminating the need for external models or datasets (SafeDecoding) and avoiding parameter updates (DINM), DSCD is positioned as a lightweight, plug-and-play solution. The introduction of two modes (MODE-1 for performance, MODE-2 for efficiency) is a smart design choice that enhances its practical applicability, allowing users to trade off precision for speed.

3.  **Comprehensive Evaluation:** The experimental section is thorough. The authors evaluate on multiple datasets (SafeEdit, AdvBench, AlpacaEval, etc.), use multiple base models (Llama2, Mistral, Qwen), and assess performance both as a standalone method and in combination with existing techniques (SFT, DPO, DINM). This provides strong evidence for the method's robustness and generalizability. The use of multiple metrics (DS, DG scores, Fluency, ASR, Harmful Score) and classifiers (RoBERTa, GPT-4o) adds credibility.

4.  **Significance of Results:** The results are impressive and support the authors' claims of state-of-the-art (SOTA) or near-SOTA performance. The key finding is that DSCD can significantly improve detoxification (e.g., +11.78% for MODE-1 alone, +4.03% when integrated with DINM) while maintaining or even improving output fluency, a common trade-off in safety interventions. The fact that it also slightly improves performance on harmless datasets like TruthfulQA is a major strength, demonstrating that the method does not degrade general capabilities.

### **Weaknesses**

1.  **Clarity of Presentation and Motivation:**
    *   **Intuition is Lacking:** The paper would be significantly stronger with a more intuitive, high-level explanation of *why* subtracting the safety layer and adding the hallucination layer from the toxic layer (Eq. 9) results in a "toxic region" distribution. The mathematical formulation is clear, but the conceptual reasoning is underdeveloped. A more detailed analogy or visualization of what each layer's distribution represents would help the reader grasp the core mechanism.
    *   **Justification for Layer Combination:** The choice to use the combination `H - S + T` feels somewhat heuristic. While the results justify its effectiveness, the paper would benefit from a deeper theoretical or empirical justification for this specific operation over other possible combinations.

2.  **Technical Depth and Analysis:**
    *   **Ablation Study is Incomplete:** The ablation study (Section 4.7) is noted, but its details are relegated to the appendix. A more prominent and detailed analysis of the individual contribution of the Toxic (T), Safety (S), and Hallucination (H) layers is crucial. Understanding which component is most critical would provide deeper insights into the method's workings.
    *   **Mechanism of "Toxic Layers":** The concept of a "toxic layer" is central but could be explored further. What kind of knowledge or processing is happening in these layers that makes them specifically responsible for toxicity? A qualitative analysis of the activations or attention patterns in these layers could provide fascinating insights.

3.  **Experimental Limitations (Acknowledged by Authors):**
    *   The authors correctly identify the limitations of not testing on a wider array of detoxification methods and newer model architectures (e.g., Llama 3). While understandable due to resource constraints, it does leave some questions about the method's universal applicability unanswered.

4.  **Potential Overstatement:** Claiming SOTA performance should be nuanced. While DSCD achieves SOTA in specific metrics and combinations (e.g., DINM+DSCD), its standalone performance is not always the absolute best across all metrics and models (e.g., in Table 3, DINM alone has a higher DG-Avg on Mistral). The paper would be more precise by clearly stating it achieves SOTA *when combined with methods like DINM* and offers a superior performance/efficiency trade-off as a standalone method.

### **Overall Assessment**

This is a strong paper with a highly innovative core idea. The proposed DSCD method represents a meaningful advance in the field of LLM safety by offering an effective, efficient, and model-intrinsic alternative to existing detoxification techniques. The comprehensive evaluation provides compelling evidence for its utility.

The main weaknesses lie in the presentation's clarity, specifically the intuitive explanation of the core algorithm, and a desire for a more in-depth ablation analysis within the main text. Despite these points, the significance of the results and the novelty of the approach make this a valuable contribution that is likely to influence future work in decoding-time safety interventions.

---

# Beyond Correctness: Rewarding Faithful Reasoning in Retrieval-Augmented Generation

Authors: Zhichao Xu, Zongyu Wu, Yun Zhou, Aosong Feng, Kang Zhou, Sangmin Woo, Kiran Ramnath, Yijun Tian, Xuan Qi, Weikang Qiu, Lin Lee Cheong, Haibo Ding

Keywords: Faithful Reasoning, Retrieval-Augmented Generation, Process Supervision, Reinforcement Learning, Agentic Search, Chain-of-Thought Faithfulness, VERITAS Framework

Comments: None

Paper link: [http://arxiv.org/abs/2510.13272v1](http://arxiv.org/abs/2510.13272v1)

## Abstract

Inspired by the success of reinforcement learning (RL) in Large Language Model (LLM) training for domains like math and code, recent works have begun exploring how to train LLMs to use search engines more effectively as tools for retrieval-augmented generation. Although these methods achieve performance improvement across QA benchmarks, many prioritize final answer correctness while overlooking the quality of intermediate reasoning steps, which may lead to chain-of-thought unfaithfulness. In this paper, we first introduce a comprehensive evaluation framework for evaluating RL-based search agents, covering three distinct faithfulness metrics: information-think faithfulness, think-answer faithfulness, and think-search faithfulness. Our evaluations reveal that a prototypical RL-based search agent, Search-R1, has significant room for improvement in this regard. To foster faithful reasoning, we introduce VERITAS (Verifying Entailed Reasoning through Intermediate Traceability in Agentic Search), a novel framework that integrates fine-grained faithfulness rewards into the reinforcement learning process. Our experiments show that models trained with VERITAS not only significantly improve reasoning faithfulness, but also achieve comparable task performance across seven QA benchmarks.

## Summary

This paper, "Beyond Correctness: Rewarding Faithful Reasoning in Retrieval-Augmented Generation," addresses a critical limitation in RL-based search agents: while these models achieve strong performance on question-answering benchmarks, they prioritize final answer correctness at the expense of **faithful reasoning** in their intermediate steps. The authors identify this as "chain-of-thought unfaithfulness," where the reasoning process does not properly align with the retrieved evidence or the final answer, making the agent's decision-making process opaque and unreliable.

The key contributions are threefold. First, the authors propose a **novel evaluation framework** that formalizes faithfulness in agentic search along three dimensions: **Information-Think faithfulness** (does the agent's reasoning use the retrieved information?), **Think-Search faithfulness** (is the search query justified by the preceding reasoning?), and **Think-Answer faithfulness** (is the final answer grounded in the reasoning?). Applying this framework to a state-of-the-art model, Search-R1, reveals a significant gap between its high task accuracy and its poor reasoning faithfulness.

Second, to bridge this gap, the authors introduce **VERITAS** (Verifying Entailed Reasoning through Intermediate Traceability in Agentic Search), a training framework that integrates these fine-grained faithfulness metrics as **process-based rewards** into the RL loop. Since using a powerful LLM-as-a-Judge for reward calculation is computationally expensive, they practically implement this by distilling a smaller, efficient **reward model** (based on Qwen2.5-14B-Instruct) to replicate the judgments, making the process scalable for RL training.

The results demonstrate that models trained with VERITAS (**VERITAS-R1**) achieve a substantial improvement in reasoning faithfulness—most notably, **Information-Think faithfulness increased by 15.3%**—while maintaining comparable, and in some cases superior, task performance across seven QA benchmarks. Crucially, the work shows that optimizing for faithful reasoning does not trade off with performance but creates a positive synergy, leading to more robust and reliable agents.

## Critique

Of course. Here is a detailed commentary on the strengths and weaknesses of the paper "Beyond Correctness: Rewarding Faithful Reasoning in Retrieval-Augmented Generation."

### Overall Assessment
This is a strong, well-executed paper that addresses a critical and timely issue in the development of AI agents. The work is compelling because it moves beyond the standard evaluation of final-answer correctness to focus on the trustworthiness and integrity of the reasoning process itself. The proposed framework, VERITAS, is a practical and effective solution, backed by rigorous experimentation.

---

### Strengths

1.  **High Novelty and Timely Contribution:**
    *   The paper identifies a clear and important gap in the literature: while RL-trained agentic search models are effective, they are often optimized for outcome-only rewards, which can lead to "reward hacking" and unfaithful reasoning chains. The focus on **process faithfulness** rather than just **outcome correctness** is a significant and forward-thinking contribution.
    *   The formalization of faithfulness into three distinct, well-motivated dimensions (Information-Think, Think-Search, Think-Answer) provides a much-needed vocabulary and framework for the community to discuss and evaluate this problem systematically.

2.  **Rigorous and Comprehensive Evaluation:**
    *   The paper begins with a strong diagnostic evaluation of an existing state-of-the-art model (Search-R1), convincingly demonstrating the problem it aims to solve. The finding that improved task performance does not guarantee faithful reasoning is a powerful motivator.
    *   The evaluation spans seven diverse QA benchmarks, covering both in-domain and out-of-domain settings, which lends strong credibility to the generalizability of the results.
    *   The use of a combination of LLM-as-a-Judge for semantic faithfulness and regex-based metrics for factual grounding is a pragmatic and effective approach.

3.  **Practical and Well-Engineered Solution:**
    *   VERITAS is not just a theoretical concept but a practical training framework. The decision to distill a large LLM judge into a smaller, efficient reward model is a crucial engineering choice that makes the approach feasible for large-scale RL training.
    *   The results are highly significant: VERITAS-R1 achieves **substantial improvements in faithfulness** (e.g., Info-Think faithfulness jumping from 0.467 to 0.842 on multi-hop QA) while **maintaining or even improving final answer accuracy**. This demonstrates a synergistic effect, showing that faithful reasoning is not a trade-off but a pathway to more robust performance.

4.  **Clarity of Presentation:**
    *   The paper is exceptionally well-structured and easy to follow. The introduction clearly outlines the problem, motivation, and contributions.
    *   Figures 1 and 3 effectively visualize the core problem and the solution's success.
    *   The writing is precise, and the pipeline (Figure 2) provides a clear overview of the VERITAS integration into the RL loop.

---

### Weaknesses

1.  **Limitations of Evaluation Metrics:**
    *   The paper correctly identifies this as a limitation. The reliance on an LLM-as-a-Judge, even a distilled one, inherits the potential biases and inconsistencies of the judge model. While the high inter-annotator agreement with human evaluation is reassuring, it's not a complete solution.
    *   The regex-based method for Think-Answer faithfulness, while precise, is brittle. It would fail to capture valid paraphrasing or logical entailment that isn't a string match, potentially misclassifying faithful answers as unfaithful.

2.  **Limited Exploration of the Think-Answer Reward:**
    *   The results show that the Think-Answer reward (`R_think-answer`) was less effective and sometimes detrimental compared to the Information-Think reward. The paper notes this but provides only a brief, speculative explanation. A deeper analysis into *why* this reward component is unstable would have been valuable. Is the metric flawed, or is it a harder objective for the RL policy to optimize?

3.  **Narrow Domain of Validation:**
    *   As acknowledged in the limitations, all experiments are conducted on open-domain QA tasks. The effectiveness of these specific faithfulness rewards in other domains like code generation, strategic planning, or creative writing is an open question. The reasoning patterns and what constitutes "faithfulness" might differ significantly.

4.  **Ablation on Reward Model Scaling:**
    *   While Appendix G touches on this, the main paper could have more explicitly discussed the cost-performance trade-off of the distilled reward model. How does the performance of VERITAS degrade if a smaller/lower-quality reward model is used? This is a key practical consideration for others looking to adopt this approach.

### Conclusion

This is a high-quality paper that makes a substantial contribution to the field of AI agents and reliable reasoning. Its strengths in identifying a critical problem, proposing a novel and practical framework, and demonstrating compelling results far outweigh its weaknesses. The work is likely to inspire further research into process-based supervision and the development of more transparent and trustworthy AI systems. The minor weaknesses primarily point to fruitful directions for future work rather than flaws in the current execution.

---

# Sparse Subnetwork Enhancement for Underrepresented Languages in Large Language Models

Authors: Daniil Gurgurov, Josef van Genabith, Simon Ostermann

Keywords: Sparse Subnetwork Enhancement, Underrepresented Languages, Large Language Models, Multilinguality, Parameter-Efficient Fine-Tuning, Neuron Identification, Language Activation Probability Entropy, Cross-lingual Alignment

Comments: preprint

Paper link: [http://arxiv.org/abs/2510.13580v1](http://arxiv.org/abs/2510.13580v1)

## Abstract

Large language models exhibit uneven performance across languages, with substantial gaps between high- and low-resource languages. We present a framework for enhancing monolingual capabilities of LLMs in underrepresented languages while preserving their general-purpose performance through targeted fine-tuning of language-specific subnetworks. Our approach identifies language-specific neurons using Language Activation Probability Entropy and fine-tunes only the weights associated with these neurons, a dedicated subnetwork, on target-language data. Experiments on Llama-3.1-8B and Mistral-Nemo-12B across 12 mid- and low-resource languages demonstrate that our method consistently outperforms full fine-tuning, FFN-only fine-tuning, LoRA adaptation, and random subset fine-tuning baselines while efficiently updating only up to 1% of model parameters. Beyond performance improvements, we observe enhanced favorable training dynamics, cross-lingual representational alignment, and systematic weight update changes. To facilitate future research, we release language-specific neuron identifications for over 100 languages as well as our adaptation pipeline, offering a cost-effective pathway for adapting state-of-the-art models to underrepresented languages.

## Summary

This paper presents "Sparse Subnetwork Enhancement for Underrepresented Languages in Large Language Models," which addresses the performance disparity between high-resource and low-resource languages in LLMs. The key contribution is a parameter-efficient framework that identifies and fine-tunes language-specific subnetworks within FFN components, enabling substantial improvements in underrepresented languages while preserving general model capabilities.

The methodology involves two main steps: first, identifying language-sensitive neurons using Language Activation Probability Entropy (LAPE), which measures how selectively neurons activate for specific languages; second, fine-tuning only the weights associated with these identified neurons (typically 0.2-1% of total parameters) using target-language data. This approach contrasts with full fine-tuning, which often causes catastrophic forgetting, and adapter-based methods that may not fully exploit the model's existing multilingual structure.

Experimental results on Llama-3.1-8B and Mistral-Nemo-12B across 12 underrepresented languages demonstrate that the method consistently outperforms full fine-tuning, FFN-only fine-tuning, LoRA adaptation, and random subset fine-tuning baselines. Notably, the approach achieves improved target-language performance while maintaining general capabilities on benchmarks like MMLU and commonsense reasoning tasks. Additional analyses reveal enhanced cross-lingual alignment in representations and systematic weight update patterns, with the most significant changes occurring in the down-projection matrices of later FFN layers. The authors release language-specific neuron identifications for over 100 languages, providing a valuable resource for future low-resource language adaptation research.

## Critique

Of course. Here is a critique of the paper "Sparse Subnetwork Enhancement for Underrepresented Languages in Large Language Models."

### Strengths

1.  **Novelty and Focused Contribution:** The paper presents a clear and compelling idea: instead of fine-tuning the entire model or using generic parameter-efficient methods like LoRA, it identifies and fine-tunes a very sparse, *language-specific* subnetwork. This builds directly on recent mechanistic interpretability findings (like LAPE) and applies them to a pressing, practical problem—improving performance for underrepresented languages. The approach is a logical and creative next step beyond existing neuron-level adaptation work.

2.  **Comprehensive and Convincing Evaluation:**
    *   **Robust Baselines:** The authors compare against a strong set of baselines, including full fine-tuning, FFN-only fine-tuning, LoRA, and a crucial **random subnetwork** baseline. This last comparison is essential for demonstrating that the *selection* of neurons (not just the sparsity) is key to the method's success.
    *   **Multi-faceted Metrics:** Performance is evaluated not only on target-language tasks (translation, comprehension, classification) but also meticulously on a suite of general-knowledge benchmarks (MMLU, Hellaswag, etc.). This dual evaluation is critical for proving the core claim of preserving general capabilities.
    *   **Scale and Reproducibility:** Experiments on two different model families (Llama and Mistral) across 12 languages lend weight to the findings. The commitment to releasing neuron identifications for over 100 languages and the adaptation pipeline is a significant contribution to the community.

3.  **Insightful Analysis:** The paper goes beyond simple performance metrics to provide valuable analysis:
    *   **Training Dynamics:** Showing that their method converges faster and more stably than baselines adds another layer of practical advantage.
    *   **Weight Change Analysis:** The finding that the `down_proj` weights in later layers undergo the most significant changes is a novel mechanistic insight that helps explain *how* the adaptation works within the model's architecture.
    *   **Representation Analysis:** Demonstrating improved cross-lingual alignment after fine-tuning is a powerful result that connects the method to broader goals in multilingual NLP.

4.  **Clarity of Presentation:** The paper is generally well-structured. The methodology section is clear, the figures are informative (e.g., Figure 1 effectively illustrates the core idea), and the results are presented in a logical flow from high-level averages to more granular analyses.

### Weaknesses

1.  **Limited Exploration of Core Hyperparameter `K%`:** A significant weakness is the lack of an ablation study on the `K%` parameter, which controls the sparsity of the subnetwork. The paper uses a fixed value of 5%, justified by prior work, but this is a central hyperparameter of their method. It is unclear:
    *   How was 5% chosen? Was it based on a validation set?
    *   How does performance vary with different `K%` values? A smaller `K%` might be even more efficient, while a larger one might capture more language-specific knowledge.
    *   Is the optimal `K%` consistent across languages? The observed variation in subnetwork sizes (Table 1) suggests it might not be.

2.  **Ambiguity in "Language-Specific" Neurons:** The concept of a "language-specific" neuron is central, but the analysis shows these neurons are not exclusive. There is overlap between related languages (e.g., Slovak and Slovenian). The paper could more deeply discuss what it means for a neuron to be "specific" to a language if it is also activated by a related one. Is it capturing a language family feature? A typological characteristic?

3.  **Speculative Explanations for Performance Variation:** The discussion of why some languages (e.g., Afrikaans) benefit more than others (e.g., Latvian) is somewhat speculative. The authors suggest it may be related to pre-training data overlap or language complexity, but they do not provide concrete evidence (e.g., by correlating improvement with the amount of pre-training data for each language). This remains an interesting open question.

4.  **Clarity in Comparison to Prior Work:** While the paper does a good job distinguishing itself from Mondal et al. (2025), the comparison could be sharper. A more direct discussion of why their approach of fine-tuning the neurons *directly* works better than adding LoRA modules *on top* of them would strengthen the narrative.

### Overall Assessment

This is a **strong and impactful paper**. It presents a novel, well-motivated, and practical method for language adaptation that is backed by extensive experiments and insightful analysis. The core strength is its elegant combination of mechanistic interpretability (LAPE) with a highly parameter-efficient fine-tuning strategy, yielding both performance gains and a preservation of general capabilities.

The primary limitations lie in the unexplored aspects of its main hyperparameter and a somewhat surface-level discussion of the linguistic reasons for performance variation. Nonetheless, the significance of the results, the release of resources, and the clarity of the core contribution make this a valuable addition to the literature on multilingual NLP and efficient model adaptation.

