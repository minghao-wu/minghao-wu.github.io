---
title: "ArXiv Daily Digest on 2025-10-30"
collection: digests
permalink: /digests/arxiv_cs_CL_2025-10-30_report
date: 2025-10-30
location: "Online"
---

Today's research landscape reveals a compelling focus on enhancing the reasoning and collaboration capabilities of large language models (LLMs), with several papers proposing innovative structural and methodological refinements. A key theme is the development of multi-agent systems and novel reasoning paradigms, such as "asynchronous thinking (AsyncThink)," which organizes internal model computations into concurrently executable structures for improved efficiency. This is complemented by data-driven frameworks for forming synergistic multi-agent teams through conversation analysis and community detection. Simultaneously, other studies address foundational training challenges, demonstrating that a simple switch from BF16 to FP16 floating-point precision can resolve the notorious training-inference mismatch in reinforcement learning (RL) fine-tuning, while another traces "value drifts" to show that a model's ethical alignment is predominantly shaped during supervised fine-tuning (SFT), with preference optimization playing a surprisingly minor role.

## TL;DR

Total papers: 58 , Selected papers: 5

**TL;DR: Recent papers focus on improving LLM reasoning and alignment through novel training paradigms, multi-agent collaboration, and addressing fundamental optimization issues.**

**Key Themes & Insights:**

1. **Reasoning Curriculum & Training Paradigms:** 
   - *Reasoning Curriculum* (https://arxiv.org/abs/2510.26143) proposes math-first RL training followed by multi-domain joint RL to bootstrap general reasoning skills
   - *AsyncThink* (https://arxiv.org/abs/2510.26658) introduces learnable asynchronous thinking protocols where LLMs organize reasoning into concurrent structures

2. **Multi-Agent Collaboration & Team Formation:**
   - *Geometry of Dialogue* (https://arxiv.org/abs/2510.26352) uses conversation graphs and community detection to automatically form synergistic LLM teams without prior model knowledge

3. **Fundamental Optimization Issues:**
   - *FP16 Precision* (https://arxiv.org/abs/2510.26788) identifies BF16 as the root cause of training-inference mismatch in RL fine-tuning, showing FP16 provides more stable optimization
   - *Value Drifts* (https://arxiv.org/abs/2510.26707) reveals SFT primarily establishes model values, with preference optimization having limited impact unless datasets contain significant value gaps

**Common Insights:** Most papers emphasize data-driven, empirically-validated approaches over complex algorithmic innovations. There's strong focus on transfer learning, with methods showing generalization to unseen tasks. The community is moving toward more transparent and mechanistic understanding of training dynamics rather than black-box optimization.

---

# Reasoning Curriculum: Bootstrapping Broad LLM Reasoning from Math

Authors: Bo Pang, Deqian Kong, Silvio Savarese, Caiming Xiong, Yingbo Zhou

Keywords: reasoning curriculum, reinforcement learning, large language models, multi-domain reasoning, cognitive skills, math-first training

Comments: 9 pages

Paper link: [http://arxiv.org/abs/2510.26143v1](http://arxiv.org/abs/2510.26143v1)

## Abstract

Reinforcement learning (RL) can elicit strong reasoning in large language models (LLMs), yet most open efforts focus on math and code. We propose Reasoning Curriculum, a simple two-stage curriculum that first elicits reasoning skills in pretraining-aligned domains such as math, then adapts and refines these skills across other domains via joint RL. Stage 1 performs a brief cold start and then math-only RL with verifiable rewards to develop reasoning skills. Stage 2 runs joint RL on mixed-domain data to transfer and consolidate these skills. The curriculum is minimal and backbone-agnostic, requiring no specialized reward models beyond standard verifiability checks. Evaluated on Qwen3-4B and Llama-3.1-8B over a multi-domain suite, reasoning curriculum yields consistent gains. Ablations and a cognitive-skill analysis indicate that both stages are necessary and that math-first elicitation increases cognitive behaviors important for solving complex problems. Reasoning Curriculum provides a compact, easy-to-adopt recipe for general reasoning.

## Summary

Based on the provided paper, here is a summary of its key contributions, methods, and results.

**Key Contributions:**
This paper introduces "Reasoning Curriculum," a novel two-stage training framework designed to enhance the general reasoning capabilities of Large Language Models (LLMs). The core idea is to first elicit strong reasoning skills in a domain well-suited for reinforcement learning (math) and then transfer and refine these skills across a broad range of other domains. The method is presented as a minimal, backbone-agnostic recipe that requires no specialized reward models, relying only on standard verifiable rewards.

**Methods:**
The proposed Reasoning Curriculum consists of two sequential stages:
1.  **Stage 1 - Math Training:** The process begins with a brief supervised fine-tuning ("Cold Start") on a small set of math problems with high-quality reasoning chains. This is followed by a math-only Reinforcement Learning (RL) phase using the DAPO objective and verifiable correctness rewards. This stage aims to robustly develop core cognitive reasoning skills like subgoal setting, enumeration, backtracking, and verification.
2.  **Stage 2 - Joint RL:** The model then undergoes RL on a mixed-domain dataset spanning six areas: Math, STEM, Code, Simulation, Logic, and Tabular tasks. This stage adapts and consolidates the reasoning skills learned in the first stage for general use. Rewards are domain-specific, using rule-based matching, model-based equivalence, or execution-based verification.

**Results:**
The method was evaluated on Qwen3-4B and Llama-3.1-8B models across a comprehensive suite of benchmarks.
*   On **Qwen3-4B**, the model trained with Reasoning Curriculum consistently outperformed similarly sized (7B) baselines and was competitive with, and sometimes exceeded, the performance of much larger (32B) models across all six domains.
*   For **Llama-3.1-8B**, a simple difficulty curriculum (medium-then-hard problems) was added to the Math-RL stage to stabilize training. The full curriculum again provided consistent performance gains over strong baselines across all domains.
*   **Ablation studies** confirmed that both the Math-RL stage and the Cold Start are necessary for achieving the full performance benefits.
*   A **cognitive-skill analysis** demonstrated that the proposed curriculum successfully increases the model's use of advanced reasoning behaviors like verification and backtracking across diverse tasks, providing mechanistic evidence for its effectiveness.

## Critique

Of course. Here is a critique of the paper "Reasoning Curriculum: Bootstrapping Broad LLM Reasoning from Math," focusing on its strengths and weaknesses.

### Strengths

1.  **Clear and Compelling Core Idea:** The central hypothesis—that math is a uniquely effective "priming" domain for eliciting general reasoning skills—is intuitive, well-motivated, and addresses a clear gap in the open-source landscape, which has been heavily focused on math and code. The proposed two-stage curriculum is simple and elegant.

2.  **Strong and Comprehensive Empirical Validation:** The paper provides extensive evidence for its claims.
    *   **Multi-domain Evaluation:** The evaluation across six distinct domains (Math, STEM, Code, Simulation, Logic, Tabular) is thorough and demonstrates the "broad" nature of the improvements.
    *   **Multi-model Validation:** Testing the approach on two different model families (Qwen and Llama) strengthens the claim that the method is "backbone-agnostic." The adaptation for Llama (using the instruct-tuned base and a difficulty curriculum) shows practical problem-solving.
    *   **Competitive Performance:** The results are impressive, particularly the Qwen3-4B model's ability to compete with or exceed much larger (32B) baseline models on several benchmarks.

3.  **Insightful Analysis:** The paper goes beyond just reporting scores.
    *   **Cognitive Skill Analysis:** The analysis of skill frequencies (subgoal, enumeration, backtracking, verification) provides a mechanistic explanation for *why* the curriculum works, linking the training process to observable behavioral changes.
    *   **Ablation Studies:** The ablations clearly demonstrate the contribution of each stage (Cold-Start, Math-RL, Joint-RL), proving that the full curriculum is necessary for optimal performance.
    *   **Stage-wise Performance Tracking:** Figure 3 is particularly revealing, showing how performance evolves (and sometimes temporarily degrades) through the curriculum stages, offering nuanced insights into skill transfer and domain-specific learning.

4.  **Practicality and Accessibility:** The emphasis on a "minimal" recipe that requires "no specialized reward models" is a significant strength. It lowers the barrier to entry for other researchers and practitioners, making the findings highly actionable and reproducible.

### Weaknesses

1.  **Limited Novelty in Constituent Parts:** While the overall *orchestration* of the curriculum is novel and valuable, the individual components are not. The use of Cold-Start SFT, Math-only RL (e.g., as in DeepSeek-R1), and Joint RL on mixed data (e.g., as in Guru) are all established techniques. The paper's primary contribution is the empirical demonstration that this specific sequence is particularly effective.

2.  **Incomplete Exploration of the "Why":** While the cognitive skill analysis is a good start, the paper could delve deeper into the underlying reasons for math's efficacy as a priming domain. Is it the structured nature of the problems? The clarity of the reward signal? The prevalence of certain reasoning patterns? A more theoretical or mechanistic discussion would strengthen the foundation of the approach.

3.  **Clarity of Presentation (Minor Issues):**
    *   **Figure 1 Caption:** The caption references a "Stage 0 (pretraining)" which is not part of the proposed 2-stage curriculum and can be slightly confusing upon first read.
    *   **Hierarchical Headings:** The nesting of sections in the provided table of contents (e.g., 4.2 being under 4.1) is non-standard and a bit awkward, though this may be an artifact of the arXiv HTML conversion.
    *   **Baseline Clarity:** While many baselines are included, a more direct comparison in the text explaining *why* the proposed method outperforms, for instance, "Guru" (which also uses multi-domain data) would be helpful. The implication is that the math-first curriculum is the key differentiator, but this could be stated more explicitly.

### Summary

This is a strong, well-executed engineering paper with a clear and impactful finding. Its main strength lies not in radical algorithmic innovation but in a smart, empirically-validated training strategy that effectively bootstraps general reasoning capabilities. The comprehensive evaluation and insightful analysis make a compelling case for the "Reasoning Curriculum." The weaknesses are primarily related to the depth of the underlying theory and the incremental nature of the components, but they do not detract significantly from the paper's practical significance and utility to the community. It provides a simple and effective recipe that is likely to be widely adopted.

---

# The Geometry of Dialogue: Graphing Language Models to Reveal Synergistic Teams for Multi-Agent Collaboration

Authors: Kotaro Furuya, Yuichi Kitagawa

Keywords: multi-agent collaboration, language model graph, community detection, synergistic teams, conversation analysis

Comments: None

Paper link: [http://arxiv.org/abs/2510.26352v1](http://arxiv.org/abs/2510.26352v1)

## Abstract

While a multi-agent approach based on large language models (LLMs) represents a promising strategy to surpass the capabilities of single models, its success is critically dependent on synergistic team composition. However, forming optimal teams is a significant challenge, as the inherent opacity of most models obscures the internal characteristics necessary for effective collaboration. In this paper, we propose an interaction-centric framework for automatic team composition that does not require any prior knowledge including their internal architectures, training data, or task performances. Our method constructs a "language model graph" that maps relationships between models from the semantic coherence of pairwise conversations, and then applies community detection to identify synergistic model clusters. Our experiments with diverse LLMs demonstrate that the proposed method discovers functionally coherent groups that reflect their latent specializations. Priming conversations with specific topics identified synergistic teams which outperform random baselines on downstream benchmarks and achieve comparable accuracy to that of manually-curated teams based on known model specializations. Our findings provide a new basis for the automated design of collaborative multi-agent LLM teams.

## Summary

Of course. Here is a summary of the paper "The Geometry of Dialogue: Graphing Language Models to Reveal Synergistic Teams for Multi-Agent Collaboration."

### Key Contribution
This paper introduces a novel, interaction-centric framework for automatically composing effective multi-agent teams of Large Language Models (LLMs) without requiring any prior knowledge of the models' internal architectures, training data, or performance on downstream tasks. The core idea is to map the latent relationships between models by analyzing the "geometry" of their conversations, thereby identifying clusters of models that are likely to collaborate synergistically.

### Method
The proposed method operates in three phases:
1.  **Conversation Generation:** Pairs of LLMs engage in structured conversations, primed with a specific topic (e.g., general, mathematics, or medicine).
2.  **Graph Construction:** A "language model graph" is built where nodes are models. The edge weight between two models is calculated as the sum of cosine similarities between the embeddings of their conversational turns, quantifying their semantic coherence. A threshold is applied to filter out weak connections.
3.  **Cluster Extraction:** Community detection algorithms (like the Louvain method) are applied to this graph to identify densely connected clusters of models, which represent promising collaborative teams.

The method is based on the assumptions that constructive dialogues occur in a coherent semantic space and that models with similar characteristics are more likely to engage in such dialogues.

### Key Results
*   **Effective Clustering:** The method successfully identified functionally coherent model clusters that aligned with their known specializations, but only when conversations were primed with a relevant topic. For example, a mathematics prompt led to a cluster containing both math-specialist models, while a medical prompt formed a distinct medical cluster.
*   **Superior Team Performance:** Teams formed from the automatically detected clusters significantly outperformed randomly assembled teams on downstream benchmarks (MMLU, GSM8K, MedQA, etc.). Crucially, their performance was comparable to, and sometimes nearly matched, that of manually-curated teams based on known model specializations, which serves as a practical upper bound.
*   **Data-Driven Team Formation:** The work demonstrates that analyzing model interactions alone is a powerful and transparent way to discover synergistic teams, providing a viable alternative to task-centric or black-box selection methods.

## Critique

Of course. Here is a commentary on the strengths and weaknesses of the paper "The Geometry of Dialogue: Graphing Language Models to Reveal Synergistic Teams for Multi-Agent Collaboration."

### Overall Assessment

This is a well-structured and compelling paper that introduces a novel, interaction-centric paradigm for a critical problem in multi-agent systems: team composition. The approach is elegant, the experimental validation is thorough, and the results are significant, demonstrating that the method works effectively without any prior knowledge of the models.

---

### Strengths

1.  **High Novelty and Conceptual Elegance:** The core idea is highly innovative. Moving away from the predominant "task-centric" view to an "interaction-centric" one is a paradigm shift. The analogy of constructing a social graph for language models, where the "conversational chemistry" defines the edges, is both intuitive and powerful. The premise that semantic coherence in dialogue reflects functional similarity is a strong, well-motivated hypothesis.

2.  **Practical and Agnostic Methodology:** A major strength is that the method requires **no prior knowledge** of model internals (architecture, training data) or performance on downstream tasks. This makes it exceptionally practical for real-world scenarios involving proprietary or poorly documented models, addressing a significant barrier in the field.

3.  **Rigorous and Multi-faceted Experimental Design:** The authors validate their method on two crucial fronts:
    *   **Qualitative Cluster Coherence:** They show that the detected communities align with known model specializations (e.g., math models cluster together when primed with a math topic), which validates that the graph is capturing meaningful latent properties.
    *   **Quantitative Performance:** They demonstrate that these automatically formed teams outperform random selection and are competitive with manually curated, type-based teams on downstream benchmarks. This moves beyond mere correlation to show a tangible performance benefit.

4.  **Insightful Analysis of Topic Priming:** The paper provides a valuable insight: the choice of conversation topic acts as a "lens" that focuses the graph on specific capabilities. This is not just a hyperparameter but a feature that allows users to steer the team formation process toward a domain of interest.

5.  **Clear and Honest Discussion of Limitations:** The paper openly addresses its key limitations, particularly computational scalability (O(N²) cost) and the potential superficiality of the "cosine similarity" metric for conversation quality. Proposing future directions like using approximate nearest neighbor search adds credibility.

---

### Weaknesses and Potential Improvements

1.  **Simplistic Collaboration Protocol:** The evaluation uses a simple majority vote for team decision-making. While sufficient to prove the core concept, it doesn't fully leverage the potential of a synergistic team. A more sophisticated protocol (e.g., multi-turn debate, reflection) could have demonstrated even greater performance gains and provided a more compelling end-to-end story. The authors acknowledge this, but it remains a limitation of the current experimental setup.

2.  **Limited Exploration of the "Relationship Value" Metric:** The use of cumulative cosine similarity, while reasonable, is arguably simplistic. It could reward repetitive or sycophantic agreement rather than truly constructive and progressive dialogue. The paper would be strengthened by a deeper analysis or ablation of this metric, perhaps comparing it to alternatives that measure topic drift or semantic progression.

3.  **Scalability is a Serious Concern:** The O(N²) cost is a major practical bottleneck. While the mention of NN-Descent is a good starting point, the paper would benefit from even preliminary results or a more detailed analysis showing how much the number of required conversations could be reduced without significant quality loss.

4.  **Clarity of Presentation (Minor):**
    *   The system prompt instructs models to be "as negative or critical as possible." The rationale for this specific choice (e.g., to avoid sycophancy and force substantive debate) could be explained more clearly, as it is a non-standard and potentially counter-intuitive design decision.
    *   The isolation of `gemma-3-1b-it` in its own community is noted but not deeply analyzed. Is this purely due to its small size, or are there other factors? A brief discussion on what the method interprets as "dissimilarity" in these cases would be insightful.

---

### Summary

**The Geometry of Dialogue** presents a novel, practical, and empirically validated method for solving the model team composition problem. Its key strength is its model-agnostic, data-driven approach that uncovers latent synergies through interaction. The results are significant, showing that automatically formed teams can compete with manually curated ones. While the collaboration protocol is simple and scalability is a concern, these are addressable limitations that do not detract from the core contribution. The paper opens a promising new research direction at the intersection of multi-agent systems and graph-based analysis.

---

# Defeating the Training-Inference Mismatch via FP16

Authors: Penghui Qi, Zichen Liu, Xiangxin Zhou, Tianyu Pang, Chao Du, Wee Sun Lee, Min Lin

Keywords: Reinforcement Learning, Training-Inference Mismatch, FP16 Precision, Large Language Models, RL Fine-Tuning, Numerical Precision, Policy Gradient, Importance Sampling

Comments: None

Paper link: [http://arxiv.org/abs/2510.26788v1](http://arxiv.org/abs/2510.26788v1)

## Abstract

Reinforcement learning (RL) fine-tuning of large language models (LLMs) often suffers from instability due to the numerical mismatch between the training and inference policies. While prior work has attempted to mitigate this issue through algorithmic corrections or engineering alignments, we show that its root cause lies in the floating point precision itself. The widely adopted BF16, despite its large dynamic range, introduces large rounding errors that breaks the consistency between training and inference. In this work, we demonstrate that simply reverting to \textbf{FP16} effectively eliminates this mismatch. The change is simple, fully supported by modern frameworks with only a few lines of code change, and requires no modification to the model architecture or learning algorithm. Our results suggest that using FP16 uniformly yields more stable optimization, faster convergence, and stronger performance across diverse tasks, algorithms and frameworks. We hope these findings motivate a broader reconsideration of precision trade-offs in RL fine-tuning.

## Summary

This paper identifies and addresses a fundamental source of instability in reinforcement learning (RL) fine-tuning of large language models (LLMs): the training-inference mismatch caused by numerical precision issues. The authors demonstrate that the widely adopted BF16 format, despite its advantages for pre-training, introduces significant rounding errors that cause divergence between the policies used during training and inference. This mismatch leads to biased gradients and a deployment gap where models optimized for training don't perform optimally during inference.

The key contribution is remarkably simple: switching from BF16 to FP16 precision effectively eliminates this mismatch. FP16's higher precision (10 mantissa bits vs. BF16's 7) creates a buffer that absorbs implementation differences between training and inference engines, preventing rounding errors from accumulating. The method requires only a few lines of code change in modern frameworks and needs no modifications to model architectures or learning algorithms.

Through extensive experiments across multiple RL algorithms (GRPO, GSPO, TIS, MIS, PG), model types (dense, MoE, LoRA), and frameworks (VeRL, Oat), the authors show that FP16 consistently delivers superior results. It provides more stable optimization, faster convergence, and stronger performance across diverse tasks. Notably, FP16 enables even simple policy gradient methods to outperform complex algorithmic corrections in BF16, while also closing the deployment gap that previous methods couldn't address. The work suggests that the precision trade-off should be reconsidered specifically for RL fine-tuning, where higher precision proves more valuable than wider dynamic range.

## Critique

Of course. Here is a detailed critique of the paper "Defeating the Training-Inference Mismatch via FP16," assessing its strengths and weaknesses.

### Overall Summary

This is a high-impact paper that presents a remarkably simple, effective, and widely applicable solution to a critical problem in LLM alignment: the instability of Reinforcement Learning (RL) fine-tuning caused by the numerical mismatch between training and inference engines. The core finding—that switching from the industry-standard BF16 to FP16 precision resolves this issue—is both surprising and powerful.

---

### Strengths

1.  **High Novelty and Counter-Intuitive Insight:** The paper's central claim is highly novel. The AI community has largely converged on BF16 as the superior format for large-scale training due to its superior dynamic range. Demonstrating that its lower precision is the root cause of a major instability issue in RL fine-tuning is a significant and counter-intuitive contribution. It reframes the problem from an algorithmic one to a numerical one.

2.  **Significance and Practical Impact:** The results are not just statistically significant; they are practically transformative. The paper shows that this simple change:
    *   **Eliminates training collapse** across a wide range of algorithms (GRPO, GSPO, PG, etc.).
    *   **Outperforms complex algorithmic fixes** (TIS, MIS) that introduce computational overhead and deployment gaps.
    *   **Generalizes extensively** across model types (dense, MoE), sizes (1.5B to 14B+), fine-tuning methods (full, LoRA), and independent frameworks (VeRL, Oat).
    This makes the contribution immediately useful to both researchers and practitioners.

3.  **Excellent Experimental Design:** The paper is exceptionally thorough in its validation.
    *   **The "Sanity Test":** The creation of a "perfectible dataset" is a clever and rigorous way to test an RL algorithm's fundamental capability, isolating it from dataset artifacts.
    *   **Comprehensive Ablations:** The ablation study in Section 4.4 cleanly disentangles the effects of training vs. inference precision, providing strong evidence for the core thesis.
    *   **Framework Comparison:** Running experiments on two independent frameworks (VeRL and Oat) robustly demonstrates that the finding is not an implementation artifact.

4.  **Clear and Compelling Presentation:** The paper is well-structured and easy to follow.
    *   The problem (training-inference mismatch, biased gradient, deployment gap) is clearly explained in the background.
    *   The explanation of *why* FP16 works (higher mantissa precision reduces rounding error accumulation) is intuitive and well-supported by the offline analysis in Section 3.5.
    *   Figures, especially the multi-panel Figure 1, are highly effective at conveying the consistency and scale of the improvement.

---

### Weaknesses

1.  **Limited Exploration of Scalability to Extreme Model Sizes:** The paper acknowledges but does not fully address the potential limitations of FP16 for "extremely large models." While results on a 14B dense model and a 30B MoE model are promising, the current industry is pushing towards models with hundreds of billions of parameters. A discussion or small-scale experiment on a model in the 50B+ parameter range would have strengthened the claim of universal applicability. The concern about FP16's smaller dynamic range leading to overflows/underflows in massive models remains a theoretical caveat.

2.  **Lack of Comparison with a "True Gold Standard":** The experiments convincingly show that FP16 is better than BF16 and superior to existing algorithmic patches. However, it would be even more compelling to include a comparison with an idealized, computationally expensive baseline, such as using FP32 for both training and inference throughout the entire pipeline. This would help quantify how much of the performance gap FP16 actually closes.

3.  **Insufficient Discussion of Hardware and Ecosystem Compatibility:** The paper could more deeply discuss the practical implications of adopting FP16. While it states the change is simple, it should note that BF16 is heavily optimized on modern AI hardware (e.g., NVIDIA Ampere+ GPUs, TPUs). Are there any performance (speed/memory) trade-offs when using FP16 instead of BF16 on these platforms? A brief note on this would be valuable for practitioners.

4.  **Mechanistic Analysis is Good, but Could Be Deeper:** The offline analysis in Section 3.5 is good, but a more in-depth analysis of *how* the precision affects the gradient signals during training could provide even deeper insight. For example, tracking the distribution of gradient norms or the condition number of the optimization landscape under BF16 vs. FP16 could reveal why one leads to collapse and the other to stability.

---

### Conclusion

This is a top-tier paper that identifies a fundamental flaw in the standard practice for RL fine-tuning and provides an elegantly simple and highly effective solution. The strength of the empirical evidence, the breadth of the validation, and the clarity of the presentation are outstanding. While it could be slightly strengthened by addressing scalability to the largest models and providing a comparison to an FP32 baseline, these are minor points in the context of its significant contribution. This work is likely to have an immediate and substantial impact on the field of LLM alignment, changing the default configuration for RL fine-tuning in many research and production environments.

---

# Value Drifts: Tracing Value Alignment During LLM Post-Training

Authors: Mehar Bhatia, Shravan Nayak, Gaurav Kamath, Marius Mosbach, Karolina Stańczak, Vered Shwartz, Siva Reddy

Keywords: Value Alignment, LLM Post-Training, Value Drifts, Supervised Fine-Tuning, Preference Optimization

Comments: None

Paper link: [http://arxiv.org/abs/2510.26707v1](http://arxiv.org/abs/2510.26707v1)

## Abstract

As LLMs occupy an increasingly important role in society, they are more and more confronted with questions that require them not only to draw on their general knowledge but also to align with certain human value systems. Therefore, studying the alignment of LLMs with human values has become a crucial field of inquiry. Prior work, however, mostly focuses on evaluating the alignment of fully trained models, overlooking the training dynamics by which models learn to express human values. In this work, we investigate how and at which stage value alignment arises during the course of a model's post-training. Our analysis disentangles the effects of post-training algorithms and datasets, measuring both the magnitude and time of value drifts during training. Experimenting with Llama-3 and Qwen-3 models of different sizes and popular supervised fine-tuning (SFT) and preference optimization datasets and algorithms, we find that the SFT phase generally establishes a model's values, and subsequent preference optimization rarely re-aligns these values. Furthermore, using a synthetic preference dataset that enables controlled manipulation of values, we find that different preference optimization algorithms lead to different value alignment outcomes, even when preference data is held constant. Our findings provide actionable insights into how values are learned during post-training and help to inform data curation, as well as the selection of models and algorithms for preference optimization to improve model alignment to human values.

## Summary

This paper, "Value Drifts: Tracing Value Alignment During LLM Post-Training," investigates how Large Language Models (LLMs) acquire and express human values throughout the post-training process, specifically during supervised fine-tuning (SFT) and preference optimization. The central contribution is the introduction and analysis of **"value drifts"**—the shifts in a model's expressed stances on value-laden topics during training.

**Key Methods:**
- The authors operationalize values by analyzing the **stances** (support, neutral, oppose) a model adopts in response to value-probing prompts from their curated evaluation set, **V-PRISM**.
- They trace value drifts using two metrics: **drift magnitude** (change in stance probability) and **drift time** (speed of change).
- Experiments are conducted on Llama-3 and Qwen-3 models of various sizes, using popular SFT datasets (WildChat, Alpaca) and preference optimization methods (PPO, DPO, SimPO) with standard datasets (UltraFeedback, HH-RLHF) and a novel **synthetic preference dataset** with a controlled "value-gap."

**Key Results:**
1.  **SFT is the primary driver of value alignment.** The SFT stage rapidly and strongly imprints a model's value profile, with different datasets (e.g., WildChat vs. Alpaca) imparting distinct stances (e.g., more neutral vs. more supportive).
2.  **Standard Preference Optimization induces minimal value drift.** When using common preference datasets like UltraFeedback, subsequent PPO, DPO, or SimPO training causes little to no change to the values set during SFT. The authors attribute this to a low "value-gap" in these datasets, where chosen and rejected responses often share similar stances.
3.  **Preference Optimization algorithms behave differently under controlled conditions.** Using their synthetic dataset with a high value-gap, the study reveals algorithm-specific behaviors:
    - **PPO** largely preserves SFT-learned values due to its KL-divergence penalty.
    - **DPO** amplifies the chosen stance if it aligns with the SFT prior but only partially shifts misaligned stances.
    - **SimPO** leads to more modest and slower value drifts compared to DPO.

In conclusion, the work demonstrates that a model's final values are predominantly determined by the SFT stage, and the effectiveness of subsequent preference optimization in reshaping values is highly contingent on the value contrast present in the preference data and the specific algorithm used. These findings offer crucial insights for designing more transparent and controlled post-training pipelines.

## Critique

Of course. Here is a critique of the paper "Value Drifts: Tracing Value Alignment During LLM Post-Training," focusing on its strengths, weaknesses, and overall contribution.

### Summary of Strengths

1.  **High Novelty and Important Research Question:** The paper tackles a critically under-explored area: the *dynamics* of how Large Language Models (LLMs) acquire values *during* training, rather than just performing a post-hoc evaluation. The concept of "value drifts" is a novel and powerful framing that allows for a more mechanistic understanding of alignment.

2.  **Rigorous and Comprehensive Experimental Design:** The methodology is a major strength. The authors systematically dissect the post-training pipeline by:
    *   Using multiple model families (Llama, Qwen) and sizes.
    *   Evaluating multiple checkpoints throughout SFT and Preference Optimization (PO).
    *   Testing different, widely-used datasets for both SFT (WildChat, Alpaca) and PO (UltraFeedback, HH-RLHF).
    *   Employing three distinct PO algorithms (PPO, DPO, SimPO).
    This multi-faceted approach allows them to isolate the effects of data, algorithms, and model architecture.

3.  **Significant and Actionable Findings:** The results are compelling and have direct implications for the field:
    *   **SFT as the Primary Value Setter:** The finding that SFT is the dominant stage for value alignment is a crucial insight that challenges the common assumption that explicit preference optimization is the main driver of a model's ethical profile.
    *   **The "Value-Gap" Hypothesis:** Identifying that standard preference datasets have a low "value-gap" (chosen and rejected responses are similar in stance) elegantly explains why PO often fails to reshape values. This provides a clear, data-centric explanation for a puzzling observation.
    *   **Algorithm-Specific Behaviors:** The controlled experiment with synthetic data reveals distinct behavioral patterns for each PO algorithm (PPO preserves, DPO amplifies, SimPO modestly shifts), offering practical guidance for practitioners.

4.  **Clarity of Presentation:** The paper is well-structured, with clear definitions of key concepts (values, stances, drift magnitude/time). The figures effectively illustrate the core findings, and the narrative logically builds from one experiment to the next.

### Summary of Weaknesses

1.  **Operationalization of "Values":** The paper's core methodological choice—using discrete stances (support/neutral/oppose) as a proxy for latent values—is a necessary simplification but also a significant limitation. It flattens the complexity of human values. For example, opposing immigration for economic reasons vs. cultural reasons are conflated into the same "oppose" stance, despite reflecting entirely different value systems. The ethics statement acknowledges this, but it remains a fundamental constraint on the depth of the analysis.

2.  **Evaluation Set Limitations:** While V-PRISM is a good starting point, its derivation from PRISM inherits a cultural and geographical skew (primarily US/UK/Europe perspectives). This means the "values" being traced are not a globally representative set, and the findings might not generalize to topics more salient in other cultures.

3.  **Dependence on GPT-4o for Evaluation:** The entire analysis relies on GPT-4o to classify the stances of model generations. This introduces a potential bias, as GPT-4o has its own baked-in values and classification tendencies. While the authors performed a small-scale human validation, a more robust inter-annotator agreement study or the use of a separately trained, open-source classifier would have strengthened the reliability of the core metric.

4.  **Limited Exploration of "Why" in SFT:** The paper convincingly shows *that* SFT sets values, but offers less insight into *why* specific datasets impart specific value profiles. A deeper analysis of the linguistic or thematic properties of WildChat (leading to neutrality) versus Alpaca (leading to support) would have been valuable.

### Overall Assessment

This is an excellent and highly significant paper. Its strengths far outweigh its weaknesses. The novelty of the "value drifts" framework and the rigor of the experimental design provide a foundational contribution to the field of AI alignment. The key findings—that SFT is paramount and that standard PO datasets lack the necessary value contrast—are actionable insights that will influence how researchers and developers approach model post-training. The weaknesses are primarily related to the inherent challenges of quantifying a complex, human concept like "values," and they are well-acknowledged by the authors, providing clear avenues for future research. The paper is a major step towards a more transparent and principled understanding of how LLMs learn to express the values we train them with.

---

# The Era of Agentic Organization: Learning to Organize with Language Models

Authors: Zewen Chi, Li Dong, Qingxiu Dong, Yaru Hao, Xun Wu, Shaohan Huang, Furu Wei

Keywords: Asynchronous Thinking, Multi-Agent Systems, Reinforcement Learning, Agentic Organization, Parallel Reasoning, Language Model Reasoning

Comments: None

Paper link: [http://arxiv.org/abs/2510.26658v1](http://arxiv.org/abs/2510.26658v1)

## Abstract

We envision a new era of AI, termed agentic organization, where agents solve complex problems by working collaboratively and concurrently, enabling outcomes beyond individual intelligence. To realize this vision, we introduce asynchronous thinking (AsyncThink) as a new paradigm of reasoning with large language models, which organizes the internal thinking process into concurrently executable structures. Specifically, we propose a thinking protocol where an organizer dynamically assigns sub-queries to workers, merges intermediate knowledge, and produces coherent solutions. More importantly, the thinking structure in this protocol can be further optimized through reinforcement learning. Experiments demonstrate that AsyncThink achieves 28% lower inference latency compared to parallel thinking while improving accuracy on mathematical reasoning. Moreover, AsyncThink generalizes its learned asynchronous thinking capabilities, effectively tackling unseen tasks without additional training.

## Summary

Here is a summary of the paper "The Era of Agentic Organization: Learning to Organize with Language Models":

**Key Contributions:**
This paper introduces *asynchronous thinking (AsyncThink)*, a new reasoning paradigm where language models learn to organize their internal thinking into concurrently executable structures. The key innovation is a thinking protocol where an LLM acts as both an *organizer* that dynamically structures the reasoning process using Fork and Join actions, and *workers* that execute sub-queries concurrently. This enables adaptive agentic organization where multiple agents collaborate to solve complex problems beyond individual capabilities.

**Methods:**
The authors propose a two-stage training procedure:
1. **Cold-start format fine-tuning**: Synthetic data is generated to teach the model the syntax of AsyncThink actions (Fork, Join, Think, Answer) through supervised fine-tuning.
2. **Reinforcement learning**: The model is further optimized using a multi-objective reward system that encourages correctness, format compliance, and thinking concurrency. The RL framework extends group relative policy optimization to handle non-sequential thought samples from both organizer and workers.

The thinking protocol operates entirely through text generation without modifying the underlying LLM architecture, allowing dynamic exploration of execution structures where sequential and parallel thinking emerge as special cases.

**Results:**
Experiments on multi-solution countdown, mathematical reasoning (AMC-23, AIME-24), and Sudoku tasks demonstrate that AsyncThink:
- Achieves higher accuracy than sequential and parallel thinking baselines (e.g., 89.0% vs 70.5% on countdown)
- Reduces critical-path latency by 28% compared to parallel thinking while maintaining competitive accuracy
- Shows strong generalization to unseen tasks (e.g., Sudoku) without additional training
- Learns to effectively distribute thinking across workers, with thinking concurrency ratios reaching ~65%

The work establishes agentic organization as a promising direction for developing more efficient and capable reasoning systems, with future directions including scaling to massive heterogeneous agents, recursive organization, and human-AI collaboration.

## Critique

Of course. Here is a critique of the paper "The Era of Agentic Organization: Learning to Organize with Language Models," focusing on its strengths and weaknesses.

### Overall Summary
This paper presents "Asynchronous Thinking (AsyncThink)," a novel paradigm for structuring LLM reasoning as a dynamic, multi-agent organization. Its core strength lies in the formulation of a learnable, text-based protocol for concurrency and the compelling empirical results demonstrating improved accuracy and reduced latency. However, the evaluation's focus on a limited set of tasks and the high complexity of the proposed system are notable weaknesses.

---

### Strengths

1.  **High Novelty in Approach:**
    *   **Learnable Organization:** The central idea of moving beyond fixed, handcrafted multi-agent workflows (like debate or simple self-correction) to a policy that is *learned* through reinforcement learning is a significant conceptual leap. It frames the problem of "how to think" as an optimizable component.
    *   **Elegant Protocol Design:** The use of a text-based `<FORK>` and `<JOIN>` protocol is clever. It allows for complex, dynamically generated execution graphs without modifying the underlying transformer architecture, making it compatible with existing LLM infrastructures.
    *   **Formalization of "Agentic Organization":** The paper does a good job of formally defining its core concepts (Agent, Agent Pool, Organization Policy) and grounding them with a clear analogy to computer systems (CPU cores, multiprocess programs). This provides a solid foundation for future work in this area.

2.  **Significant and Compelling Results:**
    *   **Accuracy & Latency Gains:** The empirical results are strong. Demonstrating a **28% reduction in latency** while simultaneously **improving accuracy** over parallel thinking baselines on mathematical reasoning is a powerful claim that addresses a key bottleneck in LLM reasoning.
    *   **Generalization is a Key Highlight:** The ability of a model trained on a "multi-solution countdown" task to perform well on Sudoku and other out-of-domain tasks is one of the most impressive findings. It suggests that the model is learning a general *skill of organization* rather than just memorizing a task-specific procedure.
    *   **Comprehensive Ablation Studies:** The paper thoroughly validates its design choices through ablations, showing the importance of both the format fine-tuning stage and the thinking concurrency reward (`R_η`). The training trajectory plots (Figure 6) effectively show how the model learns to increase parallelism over time.

3.  **Clarity of Presentation:**
    *   The paper is generally well-structured and easy to follow. The figures (especially Figures 1, 2, and 4) are excellent and intuitively explain the core concepts, the protocol, and the novel latency calculation method.
    *   The case studies in Section 4.6 and the appendix are crucial for building intuition. They move beyond abstract metrics and show *how* the model is dynamically decomposing problems, which is essential for understanding the method's value.

---

### Weaknesses

1.  **Limited Scope of Evaluation:**
    *   The tasks evaluated (Multi-Solution Countdown, Math Reasoning, Sudoku) are all highly structured, logical, and decomposable problems. It remains an open question whether AsyncThink provides similar benefits for more open-ended tasks like creative writing, complex code generation, or long-form question answering where sub-tasks are less clearly defined.
    *   The choice of a relatively small base model (Qwen3-4B) is pragmatic for research but leaves a question about how the approach scales with model size. Would a 70B+ model see the same relative benefits?

2.  **Complexity and Cost:**
    *   The proposed two-stage training pipeline (synthetic data generation -> format SFT -> RL) is highly complex and computationally expensive. This could be a barrier to adoption for many research groups.
    *   The reliance on a rule-based reward system, while effective here, may not generalize to tasks where correctness is not easily verifiable by a simple rule (e.g., "is this essay well-argued?").

3.  **Clarity and Depth on Certain Points:**
    *   **Reward Hacking:** The paper briefly mentions that the thinking concurrency reward `R_η` has a threshold `τ` to prevent the model from "hacking" it, but it doesn't provide examples or a deeper analysis of what this hacking behavior looked like during training. More detail here would be valuable.
    *   **Latency Calculation:** While the critical-path latency metric is well-defined and useful for theoretical comparison, it abstracts away real-world overheads like communication costs between GPU workers in a distributed setting. A discussion of these practical latency concerns would strengthen the paper.
    *   **Comparison to Baselines:** The parallel thinking baseline uses simple majority voting. A comparison against more sophisticated aggregation techniques (e.g., a weighted verifier or a final "judge" LLM) could provide a more rigorous benchmark.

---

### Conclusion

This is a highly novel and impactful paper that successfully introduces and validates a new paradigm for LLM reasoning. Its core strength is the shift from static to learnable organization policies, demonstrated through a clever protocol and strong empirical results, particularly the model's ability to generalize its reasoning strategy. The main weaknesses lie in the limited evaluation domain and the inherent complexity of the training framework. Nonetheless, it makes a significant contribution that is likely to inspire considerable follow-up work in multi-agent and concurrent reasoning for LLMs.

