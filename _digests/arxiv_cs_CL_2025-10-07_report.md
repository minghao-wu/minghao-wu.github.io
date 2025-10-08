---
title: "ArXiv Daily Digest on 2025-10-07"
collection: digests
type: "ArXiv daily digest"
permalink: /digests/arxiv_cs_CL_2025-10-07_report
date: 2025-10-07
location: "Online"
---

Today's research landscape showcases significant advancements in cross-lingual and reasoning capabilities, with several papers proposing innovative tokenization and prompting strategies to overcome language barriers. A standout approach is **Parallel Tokenizers**, a framework that aligns vocabularies across languages to enhance cross-lingual transfer, while **Code-Switching In-Context Learning (CSICL)** progressively transitions demonstrations from target languages to English, effectively bridging reasoning gaps in multilingual settings. In reasoning and agentic systems, **Agentic Reasoning Modules (ARM)** evolve Chain-of-Thought (CoT) reasoning by discovering specialized modules via evolutionary search, and **In-the-Flow Agentic System Optimization** introduces **Flow-based Group Refined Policy Optimization (Flow-GRPO)**, a reinforcement learning method that optimizes planning within dynamic, multi-turn interactions. Additionally, **Influence Functions** offer a causal, data-centric perspective for selecting high-quality reasoning examples, emphasizing efficiency in fine-tuning large language models (LLMs).

## TL;DR

Based on the provided papers, here is a TL;DR summary of the main themes and insights:

**Core Theme:** Improving reasoning and multilingual capabilities in language models through novel optimization and architectural approaches.

**Key Insights:**

1.  **Agentic Systems & Reasoning Optimization:** Two papers focus on optimizing agentic reasoning systems. **AgentFlow** introduces in-the-flow optimization of a modular agent framework, converting multi-turn RL into tractable single-turn updates for better tool use and planning. **ARM** discovers specialized reasoning modules through evolutionary search, enhancing the basic Chain-of-Thought (CoT) unit for superior generalization across tasks and models.

2.  **Efficient Data Selection for Reasoning:** One paper uses **Influence Functions** to causally define and select high-quality reasoning data, outperforming heuristic-based methods (like perplexity) for within-model-family fine-tuning on math tasks, though cross-model transfer remains a challenge.

3.  **Cross-Lingual Transfer & Tokenization:** Two papers address multilingual limitations. **Parallel Tokenizers** redesign vocabulary creation by aligning semantically equivalent words across languages, improving cross-lingual transfer, especially for low-resource languages. **Code-Switching ICL** uses progressive language transitions in prompts to scaffold LLMs' "latent reasoning in English," effectively bridging the translation barrier and boosting performance in target and unseen languages.

**Overall Trend:** The research moves beyond simply scaling models, focusing instead on smarter system design—optimizing the fundamental components of reasoning (agents, data, tokenization) and their interaction dynamics to achieve stronger, more efficient, and more equitable performance.

---

# Parallel Tokenizers: Rethinking Vocabulary Design for Cross-Lingual Transfer

Authors: Muhammad Dehan Al Kautsar, Fajri Koto

Keywords: Parallel Tokenizers, Vocabulary Design, Cross-Lingual Transfer, Multilingual Language Models, Low-Resource Languages, Tokenization Efficiency, Semantic Alignment

Comments: 18 pages, 25 tables, 7 figures

Paper link: [http://arxiv.org/abs/2510.06128v1](http://arxiv.org/abs/2510.06128v1)

## Abstract

Tokenization defines the foundation of multilingual language models by determining how words are represented and shared across languages. However, existing methods often fail to support effective cross-lingual transfer because semantically equivalent words are assigned distinct embeddings. For example, "I eat rice" in English and "Ina cin shinkafa" in Hausa are typically mapped to different vocabulary indices, preventing shared representations and limiting cross-lingual generalization. We introduce parallel tokenizers. This new framework trains tokenizers monolingually and then aligns their vocabularies exhaustively using bilingual dictionaries or word-to-word translation, ensuring consistent indices for semantically equivalent words. This alignment enforces a shared semantic space across languages while naturally improving fertility balance. To assess their effectiveness, we pretrain a transformer encoder from scratch on thirteen low-resource languages and evaluate it on sentiment analysis, hate speech detection, emotion classification, and sentence embedding similarity. Across all tasks, models trained with parallel tokenizers outperform conventional multilingual baselines, confirming that rethinking tokenization is essential for advancing multilingual representation learning--especially in low-resource settings.

## Summary

Based on the paper "Parallel Tokenizers: Rethinking Vocabulary Design for Cross-Lingual Transfer," here is a summary focusing on its key contributions, methods, and results:

**Key Contributions:**
The paper introduces a novel framework called **parallel tokenizers** to address fundamental limitations in multilingual language models. The core idea is to redesign tokenization to ensure semantically equivalent words across languages share the same vocabulary indices, thereby enhancing cross-lingual transfer. This approach directly tackles issues like fertility imbalance (where low-resource languages require more tokens per word) and semantic misalignment in standard multilingual tokenizers.

**Methods:**
The parallel tokenizer is constructed in three main steps:
1. **Monolingual Tokenizer as Pivot**: First, a monolingual tokenizer (e.g., English) is trained, and only its word-type vocabulary is retained.
2. **Vocabulary Alignment**: Each word in this vocabulary is translated into target languages using machine translation (e.g., Google Translate), ensuring semantic equivalence.
3. **Vocabulary Concatenation**: The translated word-type tokens are combined with a monolingually trained tokenizer for each target language, and duplicates are removed. Language identity embeddings are incorporated to disambiguate unaligned tokens and maintain language-specific signals.

The authors pretrain transformer encoders from scratch on 13 low-resource languages and evaluate against baselines like mBERT (Single-102L) and a jointly trained multilingual tokenizer (Single-13L). Tasks include sentiment analysis, hate speech detection, emotion classification, and bitext mining for cross-lingual similarity.

**Key Results:**
- **Tokenization Efficiency**: Parallel tokenizers achieve lower fertility (fewer tokens per word) and parity (better cross-lingual consistency) scores than multilingual baselines, closing the gap with monolingual tokenizers.
- **Downstream Performance**: Models using parallel tokenizers consistently outperform baselines across sequence classification tasks, especially in low-data regimes (e.g., 1–50% of training data), with an average improvement of 0.72–1.28% F1.
- **Cross-Lingual Alignment**: Visualization and bitext mining show that parallel tokenizers produce more semantically clustered representations across languages, enhancing cross-lingual transfer even with limited or zero target-language data.

In conclusion, the paper demonstrates that rethinking tokenization to enforce semantic alignment across languages is a scalable and effective strategy for improving multilingual representation learning, particularly in low-resource settings.

## Critique

Of course. Here is a critique of the paper "Parallel Tokenizers: Rethinking Vocabulary Design for Cross-Lingual Transfer," focusing on its strengths and weaknesses.

### Overall Assessment
This is a strong, well-executed paper that tackles a fundamental and often overlooked problem in multilingual NLP: the misalignment of tokenization. The proposed method is intuitive, the experimental setup is rigorous, and the results are significant, particularly for low-resource languages.

---

### Strengths

1.  **Novelty and Conceptual Clarity:** The core idea of "parallel tokenizers" is highly novel. The paper convincingly argues that standard multilingual tokenizers create artificial semantic barriers by assigning different indices to equivalent words across languages. The proposed solution—aligning vocabularies via translation to enforce shared embeddings—is both simple and powerful. The concept is explained clearly with a good motivating example ("I eat rice" vs. "Ina cin shinkafa").

2.  **Comprehensive and Rigorous Evaluation:** The paper does an excellent job of validating its claims from multiple angles:
    *   **Tokenization Quality:** It uses established metrics (fertility, parity) to show that the parallel tokenizer is more efficient and balanced than standard multilingual ones.
    *   **Downstream Performance:** It tests on a diverse set of tasks (sentiment, hate speech, emotion) across various data regimes (1% to 100%), demonstrating consistent improvements.
    *   **Representation Analysis:** The use of PCA visualization and bitext mining provides compelling, intrinsic evidence that the method genuinely improves cross-lingual semantic alignment, moving beyond just task-based metrics.

3.  **Significance for Low-Resource Languages:** The focus on languages unseen by mBERT (e.g., Amharic, Oromo, Tigrinya) is a major strength. The results show that the method is particularly beneficial in these challenging, low-resource scenarios, making it a valuable contribution for promoting linguistic equity in NLP.

4.  **Thorough Experimental Design:** The paper includes important ablations and analyses that strengthen its conclusions:
    *   The comparison of "pretraining from scratch" vs. "continual pre-training" is insightful.
    *   The experiments on limited target-language data (0% and 50%) effectively demonstrate the method's advantage in cross-lingual transfer.
    *   The comparison between multilingual and monolingual fine-tuning (Section 5.5) provides a nuanced view of the benefits of the shared semantic space.

### Weaknesses

1.  **Scalability and Practical Overhead:** The paper's primary weakness is a lack of deep discussion on the scalability of the approach. Constructing a parallel tokenizer for a new language requires:
    *   Training a monolingual tokenizer for that language.
    *   Performing word-by-word translation of the entire English word-type vocabulary.
    *   A manual filtering step (back-translation) to remove invalid translations.
    While the framework is presented as "scalable," the practical engineering and computational cost of adding dozens or hundreds of languages is non-trivial and deserves more explicit discussion.

2.  **Dependence on Machine Translation Quality:** The method's effectiveness is inherently tied to the quality of the word-level MT used for vocabulary alignment. The authors acknowledge this in the "Limitations" section, noting issues with multi-word and malformed translations. This dependency could propagate errors or cultural biases from the MT system into the model's fundamental vocabulary, a point that could be explored further.

3.  **Clarity on the "Word-Type" Filtering:** The decision to only align "word-type" tokens (excluding subwords, short words, and numbers) is pragmatic but has significant implications. It means that a substantial portion (~38%) of the final vocabulary is *not* aligned across languages. The paper could do more to analyze the impact of this—for instance, are most of the cross-lingual benefits coming from the aligned 61%, or is the unaligned portion causing interference?

4.  **Minor Presentation Issues:**
    *   The difference in learning rates for Single-13L (5e-5) and Parallel-13L (1e-4) during pre-training is mentioned but not justified. A brief explanation would strengthen the reproducibility and fairness of the comparison.
    *   While the overall structure is good, the flow between some sections (e.g., from the results to the limitations) feels slightly abrupt.

### Conclusion

This paper presents a novel and impactful approach to a core problem in multilingual NLP. The **strengths—**a clever and well-motivated idea, backed by extensive and multi-faceted experiments that clearly demonstrate its value for low-resource languages—far outweigh the **weaknesses**, which are primarily concerns about long-term scalability and practical implementation details. The work successfully makes the case that "rethinking tokenization is essential for advancing multilingual representation learning," and the proposed "parallel tokenizer" is a compelling step in that direction.

---

# Influence Functions for Efficient Data Selection in Reasoning

Authors: Prateek Humane, Paolo Cudrano, Daniel Z. Kaplan, Matteo Matteucci, Supriyo Chakraborty, Irina Rish

Keywords: Agentic Systems, Reinforcement Learning, Tool Use, Multi-turn Planning, In-the-Flow Optimization, Flow-GRPO

Comments: None

Paper link: [http://arxiv.org/abs/2510.06108v1](http://arxiv.org/abs/2510.06108v1)

## Abstract

Fine-tuning large language models (LLMs) on chain-of-thought (CoT) data shows that a small amount of high-quality data can outperform massive datasets. Yet, what constitutes "quality" remains ill-defined. Existing reasoning methods rely on indirect heuristics such as problem difficulty or trace length, while instruction-tuning has explored a broader range of automated selection strategies, but rarely in the context of reasoning. We propose to define reasoning data quality using influence functions, which measure the causal effect of individual CoT examples on downstream accuracy, and introduce influence-based pruning, which consistently outperforms perplexity and embedding-based baselines on math reasoning within a model family.

## Summary

This paper introduces **AgentFlow**, a trainable agentic system for effective planning and tool use, and proposes **Flow-based Group Refined Policy Optimization (Flow-GRPO)**, a novel reinforcement learning method for optimizing such systems. The key motivation is to address limitations in existing approaches: monolithic tool-integrated reasoning models struggle with long horizons and diverse tools, while most agentic systems remain training-free or use offline methods decoupled from live multi-turn dynamics.

The core contribution is **AgentFlow**, a framework with four specialized modules (Planner, Executor, Verifier, Generator) coordinated through an evolving memory. Unlike prior work, AgentFlow directly optimizes its Planner *in-the-flow*—within the live multi-turn interaction loop—enabling dynamic adaptation to tool outputs and verification signals. The **Flow-GRPO** algorithm tackles the long-horizon, sparse-reward credit assignment problem by broadcasting a single, verifiable final-outcome reward to every turn in the trajectory. This effectively converts multi-turn RL into a sequence of tractable single-turn policy updates, using group-normalized advantages to stabilize training.

Experiments across ten diverse benchmarks (search-intensive, agentic, mathematical, and scientific reasoning) demonstrate that AgentFlow with a 7B backbone significantly outperforms specialized baselines, achieving average accuracy gains of 14.9% on search, 14.0% on agentic, 14.5% on mathematical, and 4.1% on scientific tasks. Notably, it surpasses larger proprietary models like GPT-4o. Analyses confirm that Flow-GRPO enhances planning quality, improves tool-calling reliability, reduces error rates, and enables the autonomous discovery of effective solution pathways. The method also shows positive scaling with model size and turn budgets, and outperforms offline training approaches like supervised fine-tuning, which led to performance collapse.

## Critique

Of course. Here is a critique of the paper "In-the-Flow Agentic System Optimization for Effective Planning and Tool Use," focusing on its strengths, weaknesses, novelty, significance, and clarity.

### Overall Assessment

This is a strong, well-executed paper that presents a significant advancement in training agentic systems. The core idea—optimizing a planner *within* a live, multi-turn agentic loop—is both novel and impactful, addressing a key limitation in current systems. The empirical results are compelling and thoroughly support the authors' claims.

---

### Strengths

1.  **High Novelty in Core Approach:** The paper's primary contribution, "in-the-flow" optimization, is genuinely novel. The distinction between training a monolithic policy (the standard approach) and training a single planner module within a dynamic, multi-agent system is clear and well-motivated. The proposed **AgentFlow** framework itself, with its four specialized modules (Planner, Executor, Verifier, Generator) coordinated by an evolving memory, is a well-structured and logical design.

2.  **Innovative and Well-Motivated Algorithm (Flow-GRPO):** The **Flow-based Group Refined Policy Optimization (Flow-GRPO)** algorithm is a clever and pragmatic solution to the long-horizon, sparse-reward problem. The idea of "broadcasting" a single final-outcome reward to every turn in a trajectory, thereby converting a complex multi-turn RL problem into a series of tractable single-turn updates, is both simple and powerful. The theoretical analysis in the appendix (equivalence proof, convergence) adds rigor.

3.  **Extensive and Convincing Empirical Evaluation:** The paper is exceptionally strong on experiments.
    *   **Benchmark Diversity:** Evaluation across ten benchmarks spanning search, agentic, mathematical, and scientific reasoning provides robust evidence of generalizability.
    *   **Comprehensive Baselines:** The authors compare against a wide array of strong baselines, including base LLMs, proprietary models (GPT-4o), specialized tool-integrated RL models, and a leading training-free agentic system (AutoGen).
    *   **Significant Results:** The results are not just incremental; they show substantial performance gains (e.g., +14.9% on search, surpassing GPT-4o with a 7B model). This demonstrates the significance of the proposed method.
    *   **Thorough Analysis:** The paper goes beyond main results with insightful ablations (training strategies), efficiency analysis, scaling studies (model size, turn budget), and qualitative case studies. The finding that supervised fine-tuning (SFT) leads to "catastrophic performance collapse" is a powerful argument for the necessity of their on-policy RL approach.

4.  **Clarity of Presentation:** The paper is generally well-written and structured. The figures (especially Figures 1, 2, and 4) are effective at illustrating the core concepts, the system architecture, and the optimization process. The tables are clear and support the narrative of superior performance.

---

### Weaknesses

1.  **Computational Cost and Complexity:** A significant weakness, which is only briefly mentioned, is the computational overhead. Running on-policy rollouts of a multi-module system for training is vastly more expensive than offline training or running a monolithic model. The requirement for 8 A100 GPUs and the synchronous execution of tools (with a 500s timeout) highlight this. A more detailed discussion of the training cost and scalability would be beneficial for practitioners.

2.  **Limited Analysis of Module Interactions:** While the planner is successfully optimized, the other modules (Executor, Verifier, Generator) remain frozen. The paper does not explore whether performance could be further improved by jointly or alternatively fine-tuning these other components. The potential for co-adaptation between modules is an exciting but unexplored direction.

3.  **Ablation on the "Evolving Memory":** The evolving memory is a key component of the AgentFlow architecture, but its specific design and contribution are not deeply ablated. How crucial is its structured format? How does performance change with a simpler memory (e.g., just a concatenation of past turns)? A deeper dive into this component would strengthen the architectural claims.

4.  **Potential Overfitting to Benchmark-Specific Strategies:** The tool usage analysis (Figure 5) shows the planner learns very different strategies for different tasks (e.g., favoring Google Search for 2Wiki but Wikipedia for MedQA). While this demonstrates adaptability, it also raises the question of whether the planner is learning generalizable reasoning skills or just benchmark-specific heuristics. Testing on held-out or more open-ended tasks could address this.

5.  **Reproducibility Concerns with Reliance on GPT-4o:** The final-outcome reward is provided by an "LLM-as-judge," specifically GPT-4o. This creates a dependency on a proprietary, non-deterministic model for training, which can be a barrier to reproduction and may introduce subtle biases. The authors should discuss the potential impact of this choice and whether a more transparent reward model could be used.

---

### Summary of Novelty, Significance, and Clarity

*   **Novelty:** **High.** The core concepts of "in-the-flow" optimization of an agentic system and the Flow-GRPO algorithm for tackling long-horizon credit assignment are substantial and original contributions.
*   **Significance:** **High.** The method demonstrably pushes the state-of-the-art for tool-augmented reasoning, enabling a relatively small 7B model to outperform much larger proprietary systems and specialized baselines across a wide range of complex tasks. It provides a viable path beyond static, training-free agentic systems.
*   **Clarity:** **Good to Excellent.** The paper is clearly written and well-structured. The motivation, method, and results are communicated effectively through a combination of text, algorithms, and high-quality figures and tables.

In conclusion, this paper presents a compelling and impactful approach to a central problem in AI. Its weaknesses are primarily related to practical deployment (cost) and avenues for future work, rather than flaws in its core contributions. It is likely to influence subsequent research in agentic systems and tool-augmented LLMs.

---

# ARM: Discovering Agentic Reasoning Modules for Generalizable Multi-Agent Systems

Authors: Bohan Yao, Shiva Krishna Reddy Malay, Vikas Yadav

Keywords: Influence Functions, Data Selection, Reasoning, Chain-of-Thought, Fine-tuning, Data Pruning, Math Reasoning, Large Language Models

Comments: 29 pages, 2 figures

Paper link: [http://arxiv.org/abs/2510.05746v1](http://arxiv.org/abs/2510.05746v1)

## Abstract

Large Language Model (LLM)-powered Multi-agent systems (MAS) have achieved state-of-the-art results on various complex reasoning tasks. Recent works have proposed techniques to automate the design of MASes, eliminating the need for manual engineering. However, these techniques perform poorly, often achieving similar or inferior performance to simple baselines. Furthermore, they require computationally expensive re-discovery of architectures for each new task domain and expensive data annotation on domains without existing labeled validation sets. A critical insight is that simple Chain of Thought (CoT) reasoning often performs competitively with these complex systems, suggesting that the fundamental reasoning unit of MASes, CoT, warrants further investigation. To this end, we present a new paradigm for automatic MAS design that pivots the focus to optimizing CoT reasoning. We introduce the Agentic Reasoning Module (ARM), an agentic generalization of CoT where each granular reasoning step is executed by a specialized reasoning module. This module is discovered through a tree search over the code space, starting from a simple CoT module and evolved using mutations informed by reflection on execution traces. The resulting ARM acts as a versatile reasoning building block which can be utilized as a direct recursive loop or as a subroutine in a learned meta-orchestrator. Our approach significantly outperforms both manually designed MASes and state-of-the-art automatic MAS design methods. Crucially, MASes built with ARM exhibit superb generalization, maintaining high performance across different foundation models and task domains without further optimization.

## Summary

Based on the provided paper "Influence Functions for Efficient Data Selection in Reasoning," here is a summary of its key contributions, methods, and results.

### Summary

This paper tackles the problem of defining and selecting high-quality data for fine-tuning Large Language Models (LLMs) on reasoning tasks. While it is known that a small amount of high-quality Chain-of-Thought (CoT) data can outperform massive datasets, the definition of "quality" remains vague. The authors propose a novel method that uses **Influence Functions (IFs)** to directly measure the causal effect of individual training examples on a model's downstream reasoning accuracy.

### Key Contributions

1.  **A Causal Definition of Data Quality for Reasoning:** The core contribution is a principled, causal framework for defining data quality. Instead of relying on indirect heuristics like problem difficulty or CoT length, the authors define a training example's quality by its estimated influence on whether fine-tuning improves or degrades the model's correctness on a validation set.
2.  **Influence-Based Pruning for Reasoning:** They introduce and evaluate several data pruning strategies based on IF scores:
    *   **Correct Pruning:** Removes examples that contribute the least to correct completions.
    *   **Incorrect Pruning:** Removes examples that most strongly push the model toward incorrect completions.
    *   **Combined Pruning:** A hybrid of the two approaches.

### Methods

The methodology involves:
1.  **Scoring:** After fine-tuning a base model (e.g., LLaMA-3-8B-Instruct) on the full LIMO dataset, the authors compute the influence of each training example on validation queries from the MATH500 benchmark. The validation queries are split into two sets: those where fine-tuning improved correctness (`C`) and those where it degraded correctness (`I`).
2.  **Aggregation:** For each training example, they aggregate its influence scores across all queries in sets `C` and `I` to get final scores (`s_C`, `s_I`) and rank-based scores (`r_C`, `r_I`).
3.  **Pruning:** Data is pruned by intersecting thresholds on these scores and ranks to remove either low-benefit or high-harm examples.

### Key Results

*   **Effective Within-Model Pruning:** When the same model used for pruning (LLaMA-3-8B-Instruct) is fine-tuned on the IF-selected subsets, the method consistently matches or outperforms strong baselines like Random pruning, Mid-PPL, and RDS+ across several math reasoning benchmarks (GSM8k, OlympiadBench, AMC23).
*   **Limitation in Cross-Model Transfer:** The improvements did not consistently transfer when the data subsets selected using LLaMA-3-8B-Instruct were used to fine-tune a different model family (Qwen2.5-Math-7B-Instruct). This suggests that data quality, as defined by IFs, may be model-specific.
*   **Robustness:** The method remained competitive even with more aggressive pruning (50% of data).

In conclusion, this work successfully demonstrates that influence functions provide a powerful, causality-driven tool for data selection in reasoning tasks, though its generalizability across different model architectures remains an open challenge.

## Critique

Of course. Here is a critique of the paper "Influence Functions for Efficient Data Selection in Reasoning," focusing on its strengths and weaknesses.

### Strengths

1.  **Novelty and Conceptual Clarity:** The core idea of applying influence functions (IFs) to define and select "high-quality" reasoning data is genuinely novel and well-motivated. The paper correctly identifies a gap in the literature: while data quality is acknowledged as crucial for reasoning, existing heuristics (like CoT length or problem difficulty) are indirect proxies. The proposal to define quality based on the *causal impact* of a training example on downstream accuracy is a principled and compelling approach. The explanation of how they adapt IFs for this task—by measuring influence on subsets of validation data where fine-tuning improved (`C`) or degraded (`I`) performance—is clear and insightful.

2.  **Rigorous Methodology:** The methodology is sound and thorough. The use of both raw influence scores and rank-based scores (`s(d)` and `r(d)`) to mitigate the effect of outliers is a sophisticated touch. The three distinct pruning strategies (*Correct*, *Incorrect*, *Combined*) allow for a nuanced exploration of what constitutes a "bad" datapoint (one that is unhelpful vs. one that is actively harmful).

3.  **Significant and Honest Results:** The paper presents a significant, well-supported finding: **within the same model family, IF-based pruning consistently outperforms strong baselines like RDS+ and Mid-PPL.** The results in Table 1 are convincing and demonstrate the practical utility of the method for improving data efficiency in math reasoning fine-tuning. Crucially, the authors are transparent about the limitations, explicitly stating that the benefits do not reliably transfer across model families (Llama-3 to Qwen2.5), which is an important and honest contribution to the field.

4.  **Excellent Presentation:** The paper is exceptionally well-written and structured. The abstract and introduction effectively set up the problem and the proposed solution. Figure 1 provides an intuitive visualization of the scoring mechanism, and the description of the long-tailed distributions of influence scores is a valuable observation. The limitations section is comprehensive and points to concrete directions for future work.

### Weaknesses

1.  **Limited Empirical Scope:** The most notable weakness is the relatively narrow experimental setup, which the authors openly acknowledge.
    *   **Single Dataset:** The experiments are conducted solely on the LIMO dataset, which is itself a pre-curated, high-quality subset. This raises the question of how well the method would perform on a larger, noisier, and more diverse dataset (e.g., Open-R1), where the potential for pruning might be even greater.
    *   **Limited Baselines:** While RDS+ and Mid-PPL are strong baselines, the paper does not compare against other recent reasoning-specific selection heuristics mentioned in the related work (e.g., Select2Reason). A broader comparison would better situate the performance of IFs within the existing landscape.
    *   **Lack of Statistical Significance:** The results are from a single training run. Without multiple seeds and reported confidence intervals, it is difficult to gauge the statistical significance of the performance differences, some of which appear relatively small.

2.  **The Cross-Model Family Transfer Failure:** While the honest reporting of this negative result is a strength, the failure of the method to transfer across model families (Llama-3 to Qwen2.5) is a significant practical weakness. It suggests that data quality, as defined by IFs, might be highly model-specific. This limits the general applicability and cost-effectiveness of the approach, as it implies one would need to run the computationally expensive IF calculation for each new model family.

3.  **Computational Cost:** The paper mentions but does not deeply address the "elephant in the room": the high computational cost of calculating influence functions for large models and datasets. While approximations like EK-FAC are used, this process is still far more expensive than simple baselines like perplexity filtering or embedding similarity. The method's practicality hinges on future work developing more efficient approximations, as noted in the limitations.

### Summary

This is a strong paper that introduces a novel, principled approach to a important problem in LLM fine-tuning. Its core strength lies in its conceptual innovation and the clear, empirical demonstration that influence functions can effectively identify high-quality reasoning data *within a model family*. The primary weaknesses are related to the scope of its empirical validation and the practical limitations revealed by its inability to transfer across models and its high computational cost. Despite these, the work makes a valuable contribution by providing a new, causal perspective on data quality and setting a clear agenda for future research.

---

# Code-Switching In-Context Learning for Cross-Lingual Transfer of Large Language Models

Authors: Haneul Yoo, Jiho Jin, Kyunghyun Cho, Alice Oh

Keywords: Agentic Reasoning Modules, Multi-Agent Systems, Chain-of-Thought Reasoning, Automated MAS Design, Evolutionary Search, Reasoning Optimization

Comments: None

Paper link: [http://arxiv.org/abs/2510.05678v1](http://arxiv.org/abs/2510.05678v1)

## Abstract

While large language models (LLMs) exhibit strong multilingual abilities, their reliance on English as latent representations creates a translation barrier, where reasoning implicitly depends on internal translation into English. When this process fails, performance in non-English languages deteriorates sharply, limiting the inclusiveness of LLM-based applications. Existing cross-lingual in-context learning (X-ICL) methods primarily leverage monolingual demonstrations, often failing to mitigate this barrier and instead reinforcing it. In this work, we introduce code-switching in-context learning (CSICL), a simple yet effective prompting strategy that progressively transitions from a target language to English within demonstrations and instruction to facilitate their latent reasoning in English. By explicitly scaffolding the reasoning process through controlled code-switching, CSICL acts as an implicit linguistic bridge that enhances cross-lingual alignment and reduces reliance on the translation barrier. We conduct extensive experiments across 4 LLMs, 6 datasets, and 10 languages, spanning both knowledge-intensive and reasoning-oriented domains. Our results demonstrate that CSICL consistently outperforms X-ICL baselines, achieving gains of 3.1%p and 1.9%p in both target and unseen languages, respectively. The improvement is even more pronounced in low-resource settings, with gains of 14.7% in target and 5.3% in unseen languages. These findings establish code-switching as a principled and robust approach for overcoming the translation barrier during inference, moving LLMs toward more equitable and effective multilingual systems.

## Summary

Based on the provided paper, here is a summary of its key contributions, methods, and results:

**Key Contributions:**
This paper introduces the Agentic Reasoning Module (ARM), a novel framework that enhances the traditional Chain-of-Thought (CoT) reasoning paradigm by replacing simple textual reasoning steps with specialized, agentic modules. The core contributions are: (1) presenting ARM as an evolved, enhanced version of CoT that significantly outperforms existing multi-agent systems (MAS); (2) demonstrating ARM's superior generalization across different foundation models and task domains without requiring re-optimization; and (3) providing rigorous justification for the proposed training objectives and discovery strategy.

**Methods:**
The methodology involves a decomposable framework with two key components: a Step-Generator Module (m*) that executes individual reasoning steps, and a Meta-Policy (π*) that orchestrates these steps into a complete solution. ARM is discovered through a Reflection-Guided Evolutionary Search algorithm that performs tree search over program space, starting from a basic CoT module and evolving it using mutations informed by execution trace analysis. A crucial innovation is the use of scaffolded surrogate objectives - the step-generator is optimized within stable CoT contexts for better credit assignment, while the meta-policy is discovered using simple CoT as a computationally efficient surrogate before being transferred to work with the final ARM module.

**Results:**
The paper demonstrates that ARM consistently outperforms both manually designed MAS (like Self-Refine and LLM-Debate) and automated MAS design methods (ADAS and AFlow) across multiple complex reasoning benchmarks including AIME, HMMT, GPQA, and LiveBench. Notably, the results show that simple CoT baselines often outperform complex MAS systems, while ARM further improves upon CoT performance. When combined with the discovered meta-policy (ARM + MP), the approach achieves state-of-the-art results, with particularly strong performance gains on challenging mathematical reasoning tasks (e.g., 23.4% on AIME vs 21.9% for CoT-SC using GPT-4.1-nano). The method also shows robust performance across different foundation models including GPT-4.1-nano, GPT-4o, and LLaMA-3.3-70B.

## Critique

Of course. Here is a critique of the paper "ARM: Discovering Agentic Reasoning Modules for Generalizable Multi-Agent Systems," focusing on its strengths, weaknesses, novelty, and clarity.

### Summary

The paper proposes **ARM (Agentic Reasoning Module)**, a method to automatically discover a powerful, code-based reasoning module that acts as a single "step" in a Chain-of-Thought (CoT) process. The core idea is to evolve a simple CoT step into a sophisticated, self-contained multi-agent system through a reflection-guided evolutionary search. This discovered ARM can then be used recursively or orchestrated by a separately discovered "meta-policy" to solve complex reasoning tasks.

---

### Strengths

1.  **Novel and Insightful Core Premise:** The paper's foundational insight is powerful and timely. It correctly identifies that complex, handcrafted Multi-Agent Systems (MAS) are often outperformed by simple CoT, especially with stronger foundation models. Instead of adding more complexity to the orchestration layer, the authors pivot to improving the fundamental *reasoning unit* itself. This is a compelling and novel research direction.

2.  **Elegant and Pragmatic Methodology:** The proposed two-stage discovery process (Step-Generator `m*` and Meta-Policy `π*`) is well-designed.
    *   The **scaffolded objective** for discovering the step-generator is clever. By evaluating a candidate module `m` by plugging it into a stable baseline CoT trace, it elegantly solves the credit assignment problem and constrains the search space, making the optimization tractable.
    *   The use of a **surrogate objective** for the meta-policy (training it with the cheap `m_CoT` and deploying it with the powerful `m*`) is a practical and computationally efficient strategy, which the authors convincingly justify and validate empirically.

3.  **Strong and Comprehensive Empirical Results:** The results are a significant strength. ARM consistently outperforms a wide range of strong baselines, including simple operators (CoT, Self-Refine) and state-of-the-art automated MAS designers (ADAS, AFlow), across multiple benchmarks (AIME, HMMT, GPQA) and foundation models (GPT-4.1, GPT-4o, LLaMA). The fact that it achieves this with a single, domain-agnostically discovered module underscores its generalizability.

4.  **Effective Ablation and Analysis:** The paper goes beyond just reporting scores. The analyses in Section 7 are crucial for validating the core assumptions of the method. Showing that the step-generator ranking correlates with lower per-step error and demonstrating the successful zero-shot transfer of the meta-policy provides strong evidence for the soundness of the approach.

### Weaknesses

1.  **Significant Computational Cost:** While the surrogate training for the meta-policy saves cost, the core evolutionary search for the ARM module itself is undoubtedly computationally expensive. It involves iterative LLM calls for the "Reviewer Agent" (Critic and Designer) and repeated evaluation of candidate modules on a validation set. The paper does not quantify this cost (e.g., total GPU/API hours), which is a important practical consideration for replicating or building upon this work.

2.  **The "Black Box" Nature of the Discovered Module:** The best-performing ARM (`CriticChainOfThoughtV7`) and Meta-Policy are presented in an appendix, but the paper offers limited analysis of *what* these modules actually do and *why* they are effective. A deeper qualitative analysis of the evolutionary trajectory or the key "mutations" that led to performance gains would provide more interpretability and insight.

3.  **Limited Discussion of Limitations:** The paper could more explicitly discuss the boundaries of its approach. For instance:
    *   How does the performance scale with the complexity of the validation set used for discovery?
    *   Are there types of reasoning problems (e.g., those requiring long-horizon planning or external tool use) where this approach might struggle?
    *   The method relies on a high-quality "Reviewer Agent" LLM (`o4-mini-high`); how sensitive are the results to the capability of this designer model?

4.  **Clarity of the "Agent" Terminology:** The term "agent" is used in two distinct ways, which could cause confusion. The ARM is a "module" that is itself a "multi-agent system." However, the agents within an ARM appear to be homogeneous, role-defined LLM calls, which is different from the heterogeneous, persona-driven agents in traditional MAS like CAMEL or AutoGen. This conceptual shift is central to the paper's contribution but could be articulated more sharply.

### Assessment of Novelty, Significance, and Clarity

*   **Novelty:** **High.** The concept of evolving the core CoT step into an agentic module via programmatic search is highly novel. It represents a paradigm shift from designing complex agent orchestrations to discovering a superior, general-purpose reasoning primitive.
*   **Significance:** **High.** The results demonstrate a clear and substantial improvement over existing methods. If the computational cost can be managed, ARM provides a path toward more robust and generalizable reasoning systems, moving away from brittle, domain-specific MAS designs.
*   **Clarity:** **Good.** The paper is generally well-structured and readable. The methodology is explained with formal notation and accompanied by a clear algorithm and diagram. The primary areas for improvement are a more thorough discussion of limitations and costs, and a deeper dive into the internals of the discovered modules to enhance interpretability.

### Overall Conclusion

This is a strong paper that makes a significant contribution to the field of reasoning with LLMs. It is built on a powerful insight and is backed by a clever methodology and compelling empirical results. While questions about computational cost and interpretability remain, the work successfully establishes a new and promising direction for building more capable and generalizable AI reasoning systems.

---

# In-the-Flow Agentic System Optimization for Effective Planning and Tool Use

Authors: Zhuofeng Li, Haoxiang Zhang, Seungju Han, Sheng Liu, Jianwen Xie, Yu Zhang, Yejin Choi, James Zou, Pan Lu

Keywords: N/A

Comments: 45 pages, 12 figures. Project website:
  https://agentflow.stanford.edu/

Paper link: [http://arxiv.org/abs/2510.05592v1](http://arxiv.org/abs/2510.05592v1)

## Abstract

Outcome-driven reinforcement learning has advanced reasoning in large language models (LLMs), but prevailing tool-augmented approaches train a single, monolithic policy that interleaves thoughts and tool calls under full context; this scales poorly with long horizons and diverse tools and generalizes weakly to new scenarios. Agentic systems offer a promising alternative by decomposing work across specialized modules, yet most remain training-free or rely on offline training decoupled from the live dynamics of multi-turn interaction. We introduce AgentFlow, a trainable, in-the-flow agentic framework that coordinates four modules (planner, executor, verifier, generator) through an evolving memory and directly optimizes its planner inside the multi-turn loop. To train on-policy in live environments, we propose Flow-based Group Refined Policy Optimization (Flow-GRPO), which tackles long-horizon, sparse-reward credit assignment by converting multi-turn optimization into a sequence of tractable single-turn policy updates. It broadcasts a single, verifiable trajectory-level outcome to every turn to align local planner decisions with global success and stabilizes learning with group-normalized advantages. Across ten benchmarks, AgentFlow with a 7B-scale backbone outperforms top-performing baselines with average accuracy gains of 14.9% on search, 14.0% on agentic, 14.5% on mathematical, and 4.1% on scientific tasks, even surpassing larger proprietary models like GPT-4o. Further analyses confirm the benefits of in-the-flow optimization, showing improved planning, enhanced tool-calling reliability, and positive scaling with model size and reasoning turns.

## Summary

N/A

## Critique

N/A

