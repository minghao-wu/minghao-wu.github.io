---
title: "ArXiv Daily Digest on 2025-10-16"
collection: digests
permalink: /digests/arxiv_cs_CL_2025-10-16_report
date: 2025-10-16
location: "Online"
---

Today's research landscape showcases a significant push towards enhancing the efficiency and robustness of large language models (LLMs), with a strong emphasis on reinforcement learning (RL), multi-agent systems (MAS), and multilingual adaptation. A key trend is the retrofitting of smaller models to match or surpass the performance of their larger counterparts, as demonstrated by a 300M-parameter model achieving retrieval scores comparable to 7B models. Innovations in RL are particularly prominent, with novel frameworks like Reinforcement Learning with Supervised Reward (RLSR) reframing supervised fine-tuning (SFT) within an RL loop, and methods such as Last-Token Self-Rewarding (LaSeR) and Information Gain-based Policy Optimization (IGPO) introducing lightweight, intrinsic rewards to tackle reward sparsity in multi-turn agents. Furthermore, research is increasingly tackling the challenges of complex reasoning and subjective evaluation, evidenced by frameworks that distill MAS capabilities into single models and new benchmarks that reveal the limitations of current preference learning methods in capturing nuanced creative quality.

## TL;DR

Total papers: 81 , Selected papers: 10

Here's a TL;DR summary of the key themes and insights from these recent arXiv papers:

## Multi-Agent Systems & Complex Reasoning
Several papers explore how to enhance LLM reasoning capabilities through multi-agent collaboration and efficient training:

- **IMAGINE** (https://arxiv.org/abs/2510.14406) distills multi-agent system capabilities into a single compact model, achieving 82.7% on TravelPlanner benchmark with an 8B model
- **Explore to Evolve** (https://arxiv.org/abs/2510.14438) creates web agents that aggregate information across sources using automated data synthesis
- **Agentic RL methods** (https://arxiv.org/abs/2510.14545, https://arxiv.org/abs/2510.14967) address reward sparsity in multi-turn agents through entropy balancing and information gain rewards

## Efficient Model Training & Alignment
Multiple papers focus on improving training efficiency and alignment:

- **RLSR** (https://arxiv.org/abs/2510.14200) replaces SFT with reinforcement learning using semantic similarity rewards, outperforming traditional fine-tuning
- **LaSeR** (https://arxiv.org/abs/2510.14943) enables efficient self-rewarding using last-token probabilities only
- **Small multilingual models** (https://arxiv.org/abs/2510.14274) can match 7B performance with 300M parameters through careful data curation
- **LiRA** (https://arxiv.org/abs/2510.14466) anchors low-resource languages to English semantic space for better cross-lingual performance

## Mixture-of-Experts Optimization
- **Rewiring Experts** (https://arxiv.org/abs/2510.14853) enables data-free online adaptation of MoE routing decisions during inference

## Subjective Evaluation & Human Preferences
- **WritingPreferenceBench** (https://arxiv.org/abs/2510.14616) reveals current RLHF methods struggle with subjective writing preferences, with sequence-based reward models performing near random chance (52.7%)

**Key Insight**: There's a strong trend toward making models more efficient through better training strategies (particularly RL-based approaches), multi-agent distillation, and careful data curation rather than simply scaling model size.

---

# Rewiring Experts on the Fly:Continuous Rerouting for Better Online Adaptation in Mixture-of-Expert models

Authors: Guinan Su, Yanwu Yang, Li Shen, Lu Yin, Shiwei Liu, Jonas Geiping

Keywords: Mixture-of-Experts, Test-Time Adaptation, Router Optimization, Expert Routing, Online Adaptation, MoE Models, Test-Time Rerouting

Comments: None

Paper link: [http://arxiv.org/abs/2510.14853v1](http://arxiv.org/abs/2510.14853v1)

## Abstract

Mixture-of-Experts (MoE) models achieve efficient scaling through sparse expert activation, but often suffer from suboptimal routing decisions due to distribution shifts in deployment. While existing test-time adaptation methods could potentially address these issues, they primarily focus on dense models and require access to external data, limiting their practical applicability to MoE architectures. However, we find that, instead of relying on reference data, we can optimize MoE expert selection on-the-fly based only on input context. As such, we propose \textit{a data-free, online test-time framework} that continuously adapts MoE routing decisions during text generation without external supervision or data. Our method cycles between two phases: During the prefill stage, and later in regular intervals, we optimize the routing decisions of the model using self-supervision based on the already generated sequence. Then, we generate text as normal, maintaining the modified router until the next adaption. We implement this through lightweight additive vectors that only update router logits in selected layers, maintaining computational efficiency while preventing over-adaptation. The experimental results show consistent performance gains on challenging reasoning tasks while maintaining robustness to context shifts. For example, our method achieves a 5.5\% improvement on HumanEval with OLMoE. Furthermore, owing to its plug-and-play property, our method naturally complements existing test-time scaling techniques, e.g., achieving 6\% average gains when incorporated with self-consistency on DeepSeek-V2-Lite.

## Summary

Here is a summary of the paper "Rewiring Experts on the Fly: Continuous Rerouting for Better Online Adaptation in Mixture-of-Expert models":

**Key Contributions:** This paper introduces a novel data-free, online test-time adaptation framework specifically designed for Mixture-of-Expert (MoE) models. The primary contribution is a method that dynamically optimizes expert routing decisions during text generation without relying on external data or supervision. The framework operates through lightweight additive vectors that selectively update router logits, maintaining computational efficiency while enabling continuous model adaptation.

**Methods:** The proposed approach alternates between two phases: (1) In-Context Routing Optimization, where the model uses self-supervision from the existing context to optimize routing decisions through backpropagation, and (2) Steered Generation, where text generation proceeds normally using the adapted routing parameters. To prevent over-adaptation and reduce computational overhead, the method employs dynamic layer selection based on routing confidence scores, focusing updates on layers with the most decisive expert selection patterns. The optimization uses lightweight parameter vectors that only modify router logits in selected MoE layers.

**Results:** Experimental results across multiple reasoning benchmarks (HumanEval, MBPP, GSM8K, MATH500, MMLU) show consistent performance improvements, with gains of up to 6.7% on code generation tasks. The method outperforms both in-context learning and retrieval-based adaptation approaches while requiring no external data. Notably, it achieves these improvements with 1.6× fewer FLOPs than few-shot methods and maintains robustness to context shifts in multi-turn scenarios. The approach also demonstrates strong compatibility with existing test-time techniques, achieving additional performance gains when combined with self-consistency and in-context learning. Analysis reveals that the method works by increasing router confidence, improving expert pathway selection, and highlighting task-specific experts.

## Critique

Of course. Here is a critique of the paper "Rewiring Experts on the Fly: Continuous Rerouting for Better Online Adaptation in Mixture-of-Expert models."

### Summary

This paper introduces a novel "test-time rerouting" framework for Mixture-of-Experts (MoE) models. The core idea is to dynamically adapt the router's expert selection during inference, without any external data, by using the model's own generated context for self-supervised optimization. The method alternates between optimizing lightweight "router logit adjustment" vectors and generating text with the adapted routing, creating a continuous feedback loop.

---

### Strengths

1.  **High Novelty and Conceptual Elegance:** The core idea is highly novel. While test-time adaptation (TTA) exists for dense models and retrieval-based TTA exists for MoEs, a completely data-free, online method that treats the generation context itself as a training signal for routing is a significant conceptual leap. The analogy to "neuroplasticity" is apt and compelling.

2.  **Practicality and Efficiency:** The design choices are well-motivated for real-world deployment:
    *   **Data-Free:** Eliminates the need for and overhead of maintaining/querying a reference dataset, a major limitation of methods like C3PO.
    *   **Lightweight Updates:** Optimizing only small additive vectors for router logits, rather than the model weights themselves, is computationally efficient and reduces the risk of catastrophic forgetting or overfitting.
    *   **Selective Layer Adaptation:** The confidence-based layer selection is a smart way to focus compute and avoid destabilizing the entire model, which is validated by the ablation study showing that updating all layers is less effective.

3.  **Significant and Comprehensive Results:** The empirical evaluation is thorough and convincing.
    *   The performance improvements are consistent and meaningful across three different MoE architectures (OLMoE, Qwen1.5-MoE, DeepSeek-V2-Lite) and five diverse benchmarks (coding, math, knowledge).
    *   Outperforming strong baselines like 5-shot ICL and C3PO while being data-free is a strong result.
    *   The analysis sections (Section 6) are a major strength, providing compelling evidence *why* the method works through edit distance, expert utilization heatmaps, and entropy plots.

4.  **Excellent "Plug-and-Play" Property:** Demonstrating that the method can be seamlessly combined with existing techniques like In-Context Learning and Self-Consistency to achieve further gains is a powerful argument for its utility and adoption. The 6% average gain when combined with Self-Consistency is particularly impressive.

5.  **Robustness Analysis:** The paper proactively addresses a critical concern for deployment: robustness to context shift in multi-turn conversations. The experiment in Section 6.5 shows the method remains effective even when the context contains distracting, out-of-domain examples.

### Weaknesses

1.  **Computational Overhead is Under-Discussed:** While the paper includes an efficiency analysis (Table 3), it focuses on total FLOPs and time. The practical overhead of performing multiple optimization steps during the *prefill* phase (which is typically optimized for low latency) could be a significant barrier for real-time applications. The suggestion to run optimization during "low-load timespans" is speculative and not implemented or evaluated.

2.  **Hyperparameter Sensitivity and Generalization:** The method introduces several new hyperparameters: the number of optimization steps `T`, the learning rate, the generation interval `m` (every 128 tokens), and the layer selection ratio/strategy. While an ablation is done for layer selection, a more systematic sensitivity analysis for the other parameters (e.g., how performance changes with `T` or `m`) would strengthen the paper. The choice of 128 tokens seems somewhat arbitrary.

3.  **Limited Analysis of Failure Modes:** The paper primarily highlights successes. A brief discussion of when or why the method might fail or degrade performance would provide a more balanced view. For instance, could the self-supervised loop ever lead to "hallucinatory reinforcement" where it optimizes towards an incorrect reasoning path?

4.  **Clarity of the Optimization Procedure:** The description of the two-phase process is good, but the connection between the "routing confidence" used for layer selection and the loss function could be clearer. It seems confidence is calculated from a forward pass, then layers are selected, and then the optimization occurs on the same context. A more detailed step-by-step algorithm in the main text might help.

### Overall Assessment

This is a **high-quality, impactful paper** with a strong combination of a novel idea, rigorous experimentation, and practical design. The strengths far outweigh the weaknesses. The proposed method represents a genuine advance in making MoE models more adaptive and efficient at inference time. The presentation is generally clear, with excellent figures that aid understanding. The weaknesses are primarily areas for future work or minor clarifications rather than fundamental flaws. This work is likely to influence both academic research and the practical deployment of large MoE models.

---

# Explore to Evolve: Scaling Evolved Aggregation Logic via Proactive Online Exploration for Deep Research Agents

Authors: Rui Wang, Ce Zhang, Jun-Yu Ma, Jianshu Zhang, Hongru Wang, Yi Chen, Boyang Xue, Tianqing Fang, Zhisong Zhang, Hongming Zhang, Haitao Mi, Dong Yu, Kam-Fai Wong

Keywords: Web Agents, Information Aggregation, Automated Data Synthesis, Multi-hop Reasoning, Web Navigation, Deep Research Agents

Comments: None

Paper link: [http://arxiv.org/abs/2510.14438v1](http://arxiv.org/abs/2510.14438v1)

## Abstract

Deep research web agents not only retrieve information from diverse sources such as web environments, files, and multimodal inputs, but more importantly, they need to rigorously analyze and aggregate knowledge for insightful research. However, existing open-source deep research agents predominantly focus on enhancing information-seeking capabilities of web agents to locate specific information, while overlooking the essential need for information aggregation, which would limit their ability to support in-depth research. We propose an Explore to Evolve paradigm to scalably construct verifiable training data for web agents. Begins with proactive online exploration, an agent sources grounded information by exploring the real web. Using the collected evidence, the agent then self-evolves an aggregation program by selecting, composing, and refining operations from 12 high-level logical types to synthesize a verifiable QA pair. This evolution from high-level guidance to concrete operations allowed us to scalably produce WebAggregatorQA, a dataset of 10K samples across 50K websites and 11 domains. Based on an open-source agent framework, SmolAgents, we collect supervised fine-tuning trajectories to develop a series of foundation models, WebAggregator. WebAggregator-8B matches the performance of GPT-4.1, while the 32B variant surpasses GPT-4.1 by more than 10% on GAIA-text and closely approaches Claude-3.7-sonnet. Moreover, given the limited availability of benchmarks that evaluate web agents' information aggregation abilities, we construct a human-annotated evaluation split of WebAggregatorQA as a challenging test set. On this benchmark, Claude-3.7-sonnet only achieves 28%, and GPT-4.1 scores 25.8%. Even when agents manage to retrieve all references, they still struggle on WebAggregatorQA, highlighting the need to strengthen the information aggregation capabilities of web agent foundations.

## Summary

Based on the provided paper, here is a summary focusing on its key contributions, methods, and results:

**Key Contributions:**
This paper introduces "Explore to Evolve," a novel paradigm for automatically constructing training data to enhance web agents' information aggregation capabilities, which are crucial for deep research tasks. The primary contributions are: (1) the WebAggregatorQA dataset, comprising ~10K complex question-answer pairs requiring both information retrieval and sophisticated aggregation from diverse web sources; (2) the Explore to Evolve methodology for scalable, automated data synthesis; and (3) the WebAggregator model family, fine-tuned on this dataset, which demonstrates state-of-the-art performance.

**Methods:**
The proposed Explore to Evolve framework operates through two main phases. First, in the **Explore** phase, an agent proactively navigates the live web starting from anchor URLs, using various tools (search, file parsing, dynamic interaction) to collect heterogeneous information. Second, in the **Evolve** phase, the agent synthesizes complex QA pairs by evolving high-level aggregation logics (categorized into Element, Set, Scientific Analysis, and Temporal Reasoning) into concrete, multi-step reasoning chains grounded in the explored content. This process is followed by automated quality control, including QA alignment checks and diversity constraints, to ensure data verifiability and breadth. The resulting trajectories are used to fine-tune foundation models (based on Qwen2.5/Qwen3) within the SmolAgents framework.

**Results:**
The WebAggregator models achieve superior performance, with the 8B parameter variant matching GPT-4.1 and the 32B variant surpassing it by over 10% on GAIA-text and approaching Claude-3.7-sonnet. The human-annotated WebAggregatorQA test set proves highly challenging, with top models like Claude-3.7-sonnet and GPT-4.1 achieving only 28.3% and 25.8% accuracy, respectively, highlighting the critical gap in current agents' aggregation abilities. Analyses show that even when agents successfully retrieve all necessary references, they often fail due to complex aggregation requirements, underscoring the dataset's value and the method's effectiveness in addressing a key bottleneck in web agent research.

## Critique

Of course. Here is a critique of the paper "Explore to Evolve: Scaling Evolved Aggregation Logic via Proactive Online Exploration for Deep Research Agents."

### Strengths

1.  **Novelty and Problem Formulation:** The paper tackles a well-motivated and significant gap in web agent research: the lack of focus on **information aggregation** (synthesizing insights) compared to **information seeking** (retrieving facts). The "Explore to Evolve" paradigm is a novel and ambitious approach that frames data creation as an agent task, moving beyond static, pre-collected web pages to dynamic, real-world web exploration.

2.  **Scale and Scope of the Dataset:** The creation of **WebAggregatorQA** is a major contribution. The dataset is substantial (~10K samples), diverse (12 domains, 50K+ websites), and incorporates heterogeneous sources (text, files, dynamic web elements). The focus on complex aggregation operations (Element, Set, Scientific, Temporal) is a clear differentiator from existing datasets.

3.  **Significant Empirical Results:** The results are compelling. The performance of the **WebAggregator** models is a key strength. Demonstrating that an 8B parameter model can match GPT-4.1, and a 32B model can surpass it by a significant margin on GAIA-text, provides strong evidence for the quality of the training data. The high performance on other benchmarks (WWQA, XBench) further demonstrates the transferability and robustness of the trained models.

4.  **Challenging Benchmark:** The human-annotated **WebAggregatorQA test set** is a valuable contribution in itself. The fact that even state-of-the-art models like Claude-3.7-sonnet and GPT-4.1 achieve low scores (28.3% and 25.8% respectively) effectively proves the paper's central thesis: current agents struggle with complex aggregation, and this benchmark fills a critical void in evaluation.

5.  **Analysis and Insights:** The analysis section is strong. The breakdown of information sources and aggregation operations (Figure 6) provides a clear rationale for the benchmark's difficulty. The analysis of failure modes, specifically the cases where agents retrieved all references but still failed the task (Table 5), is a powerful and direct illustration of the aggregation challenge.

### Weaknesses

1.  **Clarity of the "Evolve" Mechanism:** While the high-level concept is clear, the precise mechanism of how the agent "evolves" aggregation logic from the 12 high-level types into concrete multi-step chains could be described with more technical detail. The process feels somewhat like a "black box" guided by a prompt. A more detailed algorithmic description or a clearer explanation of how the agent selects and composes these operations would strengthen the methodology.

2.  **Limited Comparison to Closest Works:** The paper could do a better job of quantitatively comparing the *complexity* of its tasks against the most relevant prior works (e.g., TaskCraft, WebShaper) beyond the qualitative examples in Figure 5. A quantitative analysis, perhaps showing the average number of reasoning steps or unique operations per task compared to these datasets, would more concretely justify the claim of superior complexity.

3.  **Dependence on Proprietary Models:** The data construction pipeline relies heavily on GPT-4.1 for both task synthesis and quality control. This creates a dependency on a closed-source model and may limit the reproducibility and transparency of the dataset creation process for the broader research community. The potential for biases inherited from GPT-4.1 is also not discussed.

4.  **Scalability and Cost:** The method involves running a powerful LLM agent to explore the live web for each data point, which is computationally expensive and time-consuming. While the paper demonstrates the paradigm's effectiveness, it does not address the practical scalability and cost of this approach for even larger-scale dataset creation.

5.  **Presentation and Readability:** The paper is densely packed with information, which is both a strength and a weakness. The flow between sections is sometimes abrupt, and the heavy use of cross-referencing to figures and appendices can disrupt the reading flow. Some concepts, like the exact role of the "Screenshot" tool, are mentioned but not fully explained in the main text, requiring a jump to the appendix.

### Overall Assessment

This is a **high-quality and significant paper**. It identifies a critical, under-explored problem in web agents and makes a substantial contribution by introducing a novel data creation paradigm, a large-scale and challenging dataset, and a family of powerful foundation models. The empirical results are strong and convincingly support the paper's claims. The main weaknesses lie in the clarity of certain methodological details and a reliance on proprietary infrastructure. Nonetheless, the work is likely to have a high impact, pushing the research community toward developing agents that can not only find information but truly understand and synthesize it.

---

# RLSR: Reinforcement Learning with Supervised Reward Outperforms SFT in Instruction Following

Authors: Zhichao Wang, Andy Wong, Ruslan Belkin

Keywords: Reinforcement Learning, Supervised Fine-Tuning, Instruction Following, Reward Modeling, Semantic Embeddings

Comments: None

Paper link: [http://arxiv.org/abs/2510.14200v1](http://arxiv.org/abs/2510.14200v1)

## Abstract

After the pretraining stage of LLMs, techniques such as SFT, RLHF, RLVR, and RFT are applied to enhance instruction-following ability, mitigate undesired responses, improve reasoning capability and enable efficient domain adaptation with minimal data. SFT relies on the next-token prediction objective to strengthen instruction following in a base model using a large corpus of human-labeled responses. In contrast, RFT employs a RL-based approach to adapt fine-tuned reasoning models to specific domains with limited supervision. Inspired by RFT, we propose replacing SFT with RLSR to leverage the extensive SFT dataset in an RL framework, thereby improving the base model's instruction-following ability. In RLSR, the base model generates multiple responses for each prompt, and reward scores are computed as the cosine similarity in the semantic embedding space between the generated and human-labeled responses. RLSR can be utilized in multiple ways. It can directly replace SFT, achieving superior performance on instruction-following benchmarks-for example, RLSR (SB) on Qwen-7B (INFINITY) achieved an AlpacaEval win rate of 26.34%, surpassing SFT's 21.01%. Furthermore, combining SFT and RLSR further enhances downstream task performance; Qwen-7B (INFINITY) achieved a win rate of 30.73% when trained with SFT + RLSR.

## Summary

Based on the provided paper, here is a summary focusing on its key contributions, methods, and results:

**Key Contribution:** This paper introduces **Reinforcement Learning with Supervised Reward (RLSR)**, a novel method that reframes the standard Supervised Fine-Tuning (SFT) process within a Reinforcement Learning (RL) framework. The core idea is to leverage existing large-scale SFT datasets not for direct next-token prediction, but to provide a reward signal in an RL loop, thereby enhancing the base model's instruction-following capability through exploration.

**Method:** RLSR is inspired by Reinforcement Fine-Tuning (RFT) but is distinct in its goal and application. For a given prompt, the base model generates multiple candidate responses. The key innovation is the reward function: instead of using a learned reward model or sparse correctness signals, RLSR computes a reward based on the **cosine similarity in a semantic embedding space** between a generated response and the human-labeled reference response from the SFT dataset. The authors experiment with two embedding models: a lightweight SentenceBERT (SB) and a larger Qwen-Embedding model (Qwen-EM). This reward encourages the model to produce outputs that are semantically aligned with high-quality human responses. RLSR can be used in two ways: 1) as a direct replacement for SFT, or 2) as an additional fine-tuning stage after SFT (SFT+RLSR).

**Key Results:** The authors conduct extensive experiments on models like Llama-8B, Qwen-7B, and Qwen-32B using the TULU and INFINITY datasets, evaluated across 18 diverse benchmarks.
- **RLSR vs. SFT:** RLSR consistently outperforms standard SFT across most benchmarks. For example, on Qwen-7B with the INFINITY dataset, RLSR (SB) achieved an AlpacaEval win rate of **26.34%**, significantly surpassing SFT's **21.01%**.
- **SFT+RLSR:** The combined pipeline yields the strongest performance on generative instruction-following tasks. The same Qwen-7B model reached a peak AlpacaEval win rate of **30.73%** with SFT+RLSR.
- **Trade-off:** The paper notes a trade-off: RLSR-alone often excels in discriminative reasoning tasks, while SFT+RLSR is better for preserving linguistic fluency and generation quality in open-ended tasks.
- **Reward Model Scale:** The performance gain from using a larger embedding model (Qwen-EM) over a smaller one (SB) is more pronounced for larger base models (e.g., Qwen-32B).

In conclusion, RLSR demonstrates that leveraging SFT data within an RL framework, using a simple yet effective semantic similarity reward, can significantly enhance LLM alignment and instruction-following performance beyond the limits of traditional teacher-forcing fine-tuning.

## Critique

Of course. Here is a critique of the paper "RLSR: Reinforcement Learning with Supervised Reward Outperforms SFT in Instruction Following," focusing on its strengths and weaknesses.

### **Strengths**

1.  **Clear and Motivated Problem:** The paper identifies a clear gap in the standard LLM training pipeline: the Supervised Fine-Tuning (SFT) stage is a purely "teacher-forcing" method that lacks exploration, while Reinforcement Learning (RL) methods are typically reserved for later stages (RLHF/RLVR). The core question—"Can we replace SFT with an RL method that uses the same data?"—is well-motivated and significant.

2.  **Novelty of the Approach:** The proposed method, RLSR, is a novel and elegant adaptation of the RFT framework. Its key innovation is using a simple, unsupervised reward function based on the cosine similarity between the embeddings of a generated response and the human reference response. This allows it to leverage large-scale SFT datasets within an RL paradigm, directly targeting the instruction-following objective.

3.  **Comprehensive and Rigorous Evaluation:** This is a major strength of the paper. The authors conduct extensive experiments across:
    *   **Multiple Model Scales:** Llama-8B, Qwen-7B, Qwen-32B.
    *   **Multiple High-Quality Datasets:** TULU and INFINITY.
    *   **A Wide Array of Benchmarks:** 18 diverse tasks from the LM Evaluation Harness, plus AlpacaEval and Arena-Hard for open-ended generation.
    *   **Multiple Configurations:** SFT, RLSR (with two different reward models), and the hybrid SFT+RLSR.
    This thoroughness makes the results highly convincing and demonstrates the robustness of the approach.

4.  **Significant and Nuanced Results:** The results are not just positive; they are insightful. The paper shows that:
    *   **RLSR can directly replace SFT** and achieve superior performance on most benchmarks.
    *   **SFT+RLSR is often the strongest approach** for generative tasks, showing the methods are complementary.
    *   The choice of reward model (lightweight vs. heavy) has a **scale-dependent effect**, which is a valuable practical insight.
    The improvement on AlpacaEval (e.g., 21.01% → 30.73% for SFT+RLSR on Qwen-7B) is substantial and clearly demonstrates the practical significance of the method.

### **Weaknesses**

1.  **Computational Cost and Efficiency:** The paper explicitly states that RLSR consumes more FLOPs than SFT due to its need for multiple rollouts per prompt. However, it does not provide a quantitative comparison of the computational cost, training time, or memory footprint. For practitioners, understanding this trade-off (performance gain vs. cost) is critical. The claim that it is "more computationally expensive" needs to be substantiated with data.

2.  **Clarity on the "Why":** While the paper excellently demonstrates *that* RLSR works, it provides less insight into *why* it works. The authors hypothesize that the RL framework's exploration leads to better generalization beyond the token-level mimicry of SFT. However, a deeper analysis is missing. For instance:
    *   What kinds of responses is RLSR rewarding that SFT would not produce? Qualitative examples would be very illuminating.
    *   Does RLSR lead to more semantically correct but lexically diverse responses compared to the reference?
    A discussion or analysis along these lines would strengthen the paper significantly.

3.  **Presentation and Readability:**
    *   **Figure 1:** The schematic in Figure 1 is helpful, but its low resolution in the provided text makes it difficult to read the labels and fully understand the flow.
    *   **Data Overload:** The presentation of results across three massive tables (Tables 1-3) with 18 metrics each is comprehensive but overwhelming. The key takeaways are buried in the text of Section 3.2. The paper would benefit from a summarized, high-level table or a focused analysis on a curated subset of the most representative benchmarks, directing readers to the appendix for the full data.

4.  **Limited Discussion of Failure Modes:** The paper focuses on where RLSR succeeds but gives less attention to where it underperforms SFT or SFT+RLSR (e.g., on some tasks for Qwen-32B). A brief discussion of potential failure modes or the limitations of the cosine similarity reward in certain contexts (e.g., for tasks where lexical precision is more important than semantic similarity) would provide a more balanced view.

### **Overall Assessment**

This is a strong paper that presents a novel, well-evaluated, and effective method for improving LLM instruction-following. The core idea is simple yet powerful, and the extensive empirical evidence makes a compelling case for its adoption. The main weaknesses lie in the lack of computational cost analysis and a deeper mechanistic explanation for its success. Nonetheless, the significance of the results and the clarity of the central thesis make it a valuable contribution to the field of LLM alignment and fine-tuning.

---

# IMAGINE: Integrating Multi-Agent System into One Model for Complex Reasoning and Planning

Authors: Xikai Zhang, Bo Wang, Likang Xiao, Yongzhi Li, Quan Chen, Wenju Wu, Liu Liu

Keywords: Multi-Agent Systems, Complex Reasoning and Planning, Knowledge Distillation, Reinforcement Learning, Travel Planning, Language Model Training

Comments: None

Paper link: [http://arxiv.org/abs/2510.14406v1](http://arxiv.org/abs/2510.14406v1)

## Abstract

Although large language models (LLMs) have made significant strides across various tasks, they still face significant challenges in complex reasoning and planning. For example, even with carefully designed prompts and prior information explicitly provided, GPT-4o achieves only a 7% Final Pass Rate on the TravelPlanner dataset in the sole-planning mode. Similarly, even in the thinking mode, Qwen3-8B-Instruct and DeepSeek-R1-671B, only achieve Final Pass Rates of 5.9% and 40%, respectively. Although well-organized Multi-Agent Systems (MAS) can offer improved collective reasoning, they often suffer from high reasoning costs due to multi-round internal interactions, long per-response latency, and difficulties in end-to-end training. To address these challenges, we propose a general and scalable framework called IMAGINE, short for Integrating Multi-Agent System into One Model. This framework not only integrates the reasoning and planning capabilities of MAS into a single, compact model, but also significantly surpass the capabilities of the MAS through a simple end-to-end training. Through this pipeline, a single small-scale model is not only able to acquire the structured reasoning and planning capabilities of a well-organized MAS but can also significantly outperform it. Experimental results demonstrate that, when using Qwen3-8B-Instruct as the base model and training it with our method, the model achieves an 82.7% Final Pass Rate on the TravelPlanner benchmark, far exceeding the 40% of DeepSeek-R1-671B, while maintaining a much smaller model size.

## Summary

Here is a summary of the paper "IMAGINE: Integrating Multi-Agent System into One Model for Complex Reasoning and Planning":

**Key Contributions:**
The paper introduces IMAGINE, a framework that integrates the capabilities of a Multi-Agent System (MAS) into a single, compact language model. This approach addresses critical limitations of traditional MAS, including high inference costs from multi-turn interactions, long latency, and difficulties in end-to-end training. The method enables a small model (8B parameters) to not only replicate but significantly surpass the performance of carefully designed MAS while being more efficient and deployable.

**Methods:**
IMAGINE employs a three-stage pipeline:
1. **New Query Generation**: Expands dataset diversity by creating 4,105 new queries for the TravelPlanner benchmark using GPT-4o, ensuring varied complexity and constraints.
2. **Multi-Agent System-based Inference Data Generation**: Uses a custom MAS with three agents (Reasoner, Judge, Reflector) to generate high-quality reasoning traces. The MAS performs reflection and error correction, producing structured training data that captures collaborative reasoning.
3. **Agentic Reasoning Training**: 
   - **Agentic SFT**: Supervised fine-tuning on MAS-generated data to distill collective reasoning into a single model.
   - **Agentic GRPO**: Reinforcement learning with a custom rule-based reward function that checks format compliance, constraint satisfaction, and reflection quality, further enhancing reasoning capabilities.

**Results:**
On the TravelPlanner benchmark (1,000 test queries), IMAGINE achieves:
- **82.7% Final Pass Rate**, drastically outperforming strong baselines like DeepSeek-R1-671B (40%) and GPT-4o (7%).
- Significant improvements across all metrics: Commonsense Constraint Macro Pass Rate (92.5%, +32.2% over MAS) and Hard Constraint Macro Pass Rate (86.9%, +19.1%).
- The trained 8B model is highly efficient, eliminating multi-turn interactions and reducing inference costs and latency compared to MAS.

This work demonstrates that a single model can effectively learn and exceed the collaborative reasoning of multi-agent systems, offering a scalable and practical solution for complex planning tasks.

## Critique

Of course. Here is a critique of the paper "IMAGINE: Integrating Multi-Agent System into One Model for Complex Reasoning and Planning," covering its strengths, weaknesses, and overall presentation.

### Summary of Strengths

1.  **Strong and Significant Results:** The paper's most compelling strength is the empirical evidence. Achieving an **82.7% Final Pass Rate** on the challenging TravelPlanner benchmark, starting from a base model (Qwen3-8B-Instruct) that scored only 5.9%, is a remarkable result. The fact that this performance significantly surpasses not only the base model but also a much larger model (DeepSeek-R1-671B at 40%) and their own carefully-designed Multi-Agent System (45.8%) provides powerful validation for the proposed method.

2.  **Addresses a Real and Practical Problem:** The work effectively identifies and tackles the key limitations of LLM-based Multi-Agent Systems (MAS): high computational/inference costs, long latency due to multi-turn interactions, and difficulty in end-to-end training. Proposing a method to "distill" the capabilities of a MAS into a single, efficient model is a highly practical and valuable contribution.

3.  **Clear and Well-Structured Methodology:** The three-stage pipeline (Query Generation → MAS Inference Data Generation → Agentic Reasoning Training) is logically sound and clearly explained. The paper does a good job of walking the reader through each step, including the rationale for data generation and the specific architecture of their MAS (Reasoner, Judge, Reflector).

4.  **Comprehensive Evaluation:** The use of the full suite of TravelPlanner metrics (Delivery Rate, various Constraint Pass Rates, Final Pass Rate) provides a thorough assessment of performance. The inclusion of an extensive set of baselines, including various prompting strategies and state-of-the-art models, strengthens the credibility of the claims.

### Summary of Weaknesses

1.  **Limited Novelty in Core Concept:** The core idea of "distillation" – training a smaller model to mimic the behavior of a larger, more powerful system (the MAS in this case) – is a well-established technique in machine learning. The primary novelty lies in its application to distilling the collaborative, multi-step reasoning process of a MAS for complex planning tasks, rather than just the final outputs. While valuable, the paper could more explicitly position its novelty against prior knowledge distillation and imitation learning literature.

2.  **Heavy Reliance on a Single Benchmark:** The entire methodology and evaluation are centered on the TravelPlanner dataset. While it is a recognized and challenging benchmark for planning, the generalizability of the IMAGINE framework remains unproven. The paper would be significantly stronger if it demonstrated the approach's effectiveness on other complex reasoning tasks (e.g., mathematical reasoning, code generation, or other planning domains like logistics).

3.  **Cost and Complexity of the Data Generation Pipeline:** The proposed method relies on a computationally expensive and complex data generation phase. It requires running a MAS that involves multiple large models (DeepSeek-R1-671B, two "Judge" models, and Gemini-2.5-Flash) for thousands of queries. This upfront cost is substantial and could be a barrier for wider adoption, somewhat offsetting the later inference efficiency gains.

4.  **Opaque "Reflection" Mechanism:** The "Reflection Check" in the custom GRPO reward function, which gives a +0.5/-0.5 reward based on the presence of a reflection, feels somewhat simplistic and heuristic. The paper shows a correlation between the model claiming "no error" and improved performance (Figure 9), but it doesn't deeply analyze the *quality* of these reflections or prove that this reward component is causally driving the improvement.

### Clarity of Presentation

The paper is generally well-written and clearly structured. The figures are helpful in visualizing the framework (Figure 2), the MAS process (Figure 4), and the reward function (Figure 5). The results are presented in a clear table and supporting graphs.

**Areas for Improvement in Presentation:**
*   **Figure 1:** The caption "Final Pass Rate under different models" is too vague. The y-axis is unlabeled, and it's unclear what the specific numbers are. A proper, labeled bar chart in the main text would be more informative.
*   **Repetition:** The introduction and related work sections have some repetitive statements about the performance of baseline models on TravelPlanner. This could be condensed.

### Overall Assessment

This is a strong paper that presents a highly effective and practical framework for enhancing complex reasoning in smaller language models. The results are impressive and directly address significant limitations of current Multi-Agent Systems. The primary weaknesses are the limited demonstration of generalizability beyond a single task and the non-trivial cost of the data generation process. Despite these points, the work makes a compelling case for the "distillation of collaborative reasoning" as a powerful paradigm and represents a valuable contribution to the field. If the approach generalizes to other domains, it could have a substantial impact on the deployment of efficient, high-performance reasoning models.

---

# Agentic Entropy-Balanced Policy Optimization

Authors: Guanting Dong, Licheng Bao, Zhongyuan Wang, Kangzhi Zhao, Xiaoxi Li, Jiajie Jin, Jinghan Yang, Hangyu Mao, Fuzheng Zhang, Kun Gai, Guorui Zhou, Yutao Zhu, Ji-Rong Wen, Zhicheng Dou

Keywords: Agentic Reinforcement Learning, Entropy-Balanced Policy Optimization, Web Agents, Tool Learning, Multi-turn Interaction, Rollout Sampling Diversity, Gradient Clipping

Comments: Working in progress

Paper link: [http://arxiv.org/abs/2510.14545v1](http://arxiv.org/abs/2510.14545v1)

## Abstract

Recently, Agentic Reinforcement Learning (Agentic RL) has made significant progress in incentivizing the multi-turn, long-horizon tool-use capabilities of web agents. While mainstream agentic RL algorithms autonomously explore high-uncertainty tool-call steps under the guidance of entropy, excessive reliance on entropy signals can impose further constraints, leading to the training collapse. In this paper, we delve into the challenges caused by entropy and propose the Agentic Entropy-Balanced Policy Optimization (AEPO), an agentic RL algorithm designed to balance entropy in both the rollout and policy update phases. AEPO comprises two core components: (1) a dynamic entropy-balanced rollout mechanism that adaptively allocate global and branch sampling budget through entropy pre-monitoring, while imposing a branch penalty on consecutive high-entropy tool-call steps to prevent over-branching issues; and (2) Entropy-Balanced Policy Optimization that inserts a stop-gradient operation into the high-entropy clipping term to preserve and properly rescale gradients on high-entropy tokens, while incorporating entropy-aware advantage estimation to prioritize learning on high-uncertainty tokens. Results across 14 challenging datasets show that AEPO consistently outperforms 7 mainstream RL algorithms. With just 1K RL samples, Qwen3-14B with AEPO achieves impressive results: 47.6% on GAIA, 11.2% on Humanity's Last Exam, and 43.0% on WebWalker for Pass@1; 65.0% on GAIA, 26.0% on Humanity's Last Exam, and 70.0% on WebWalker for Pass@5. Further analysis reveals that AEPO improves rollout sampling diversity while maintaining stable policy entropy, facilitating scalable web agent training.

## Summary

Based on the provided paper, here is a summary focusing on its key contributions, methods, and results:

**Summary**

The paper introduces **Agentic Entropy-Balanced Policy Optimization (AEPO)**, a novel reinforcement learning (RL) algorithm designed to train more effective and stable web agents based on large language models (LLMs). The core motivation stems from identifying two key challenges in existing "agentic RL" methods that overly rely on entropy signals to guide exploration: (1) **High-Entropy Rollout Collapse**, where LLMs over-branch on a few trajectories with consecutive high-entropy tool-call steps, limiting sampling diversity, and (2) **High-Entropy Token Gradient Clipping**, where standard RL algorithms clip the gradients of high-entropy tokens during policy updates, hindering the learning of exploratory behaviors.

**Key Contributions & Methods**

1.  **Dynamic Entropy-Balanced Rollout:** To mitigate rollout collapse, AEPO first employs an **entropy pre-monitoring** phase. This dynamically allocates the sampling budget between global trajectory sampling and partial branch sampling based on the entropy gap between the initial question and the average tool-call entropy. It then uses an **entropy-balanced adaptive rollout** that penalizes consecutive high-entropy branching, preventing over-exploration on specific paths and promoting diversity.

2.  **Entropy-Balanced Policy Optimization:** To address gradient clipping, AEPO incorporates a **stop-gradient operation** into the high-entropy clipping term during policy updates. This preserves and rescales the gradients of valuable high-entropy tokens, allowing the model to learn exploratory behaviors. It also introduces **entropy-aware advantage estimation**, which reshapes the advantage function to prioritize learning on high-uncertainty tokens, further encouraging exploration.

**Key Results**

The authors evaluated AEPO across 14 challenging benchmarks covering deep information seeking, knowledge-intensive reasoning, and computational reasoning tasks.

*   **State-of-the-Art Performance:** AEPO consistently outperformed 7 mainstream RL baselines (including GRPO, ARPO, and clipping-optimized methods). With only 1K training samples, a Qwen3-14B model equipped with AEPO achieved impressive results, such as **47.6% Pass@1 on GAIA**, **11.2% Pass@1 on Humanity's Last Exam (HLE)**, and **43.0% Pass@1 on WebWalkerQA**.
*   **Improved Sampling and Efficiency:** Analysis showed that AEPO significantly improves rollout sampling diversity and achieves higher Pass@5 scores (e.g., 65.0% on GAIA). It also demonstrated greater **tool-call efficiency**, requiring roughly half the number of tool invocations compared to other RL algorithms to achieve superior performance.
*   **Enhanced Training Stability:** AEPO maintained more stable policy entropy dynamics throughout RL training compared to other methods, which often suffered from entropy collapse, leading to more robust and scalable web agent training.

## Critique

Here's a balanced assessment of the paper "Agentic Entropy-Balanced Policy Optimization":

**Strengths:**

**Novelty and Technical Contribution:**
- The paper identifies and systematically addresses two specific entropy-related problems in agentic RL that haven't been thoroughly studied: "High-Entropy Rollout Collapse" and "High-Entropy Token Gradient Clipping"
- The proposed AEPO framework offers a comprehensive solution with two complementary components addressing both rollout and policy optimization phases
- The entropy pre-monitoring mechanism for dynamic resource allocation is theoretically grounded in information bottleneck theory
- The integration of stop-gradient operations for high-entropy token preservation is technically sophisticated

**Experimental Evaluation:**
- Extensive evaluation across 14 diverse datasets covering deep information seeking, knowledge-intensive reasoning, and computational reasoning
- Strong empirical results showing consistent improvements over 7 baseline RL algorithms
- The reported performance gains are substantial (e.g., 47.6% on GAIA, 11.2% on HLE with only 1K training samples)
- Comprehensive ablation studies and analysis including diversity analysis, tool-call efficiency, and entropy stability

**Presentation and Methodology:**
- Clear problem formulation with well-defined mathematical notation
- Detailed algorithmic descriptions with pseudocode
- Good visualization of the core concepts and experimental results
- The paper follows a logical structure from problem identification to solution proposal and validation

**Weaknesses:**

**Technical Concerns:**
- The theoretical foundation for some design choices could be more rigorous (e.g., the linear penalty function P^(l) appears somewhat arbitrary)
- The relationship between entropy and actual exploration quality isn't thoroughly established - high entropy doesn't necessarily guarantee useful exploration
- The computational overhead of the entropy pre-monitoring phase isn't quantified

**Experimental Limitations:**
- While the number of datasets is impressive, the paper doesn't sufficiently address the computational cost and training time compared to baselines
- Limited analysis of failure cases or scenarios where AEPO might underperform
- The comparison with larger models (like GPT-4o, DeepSeek-R1-671B) is somewhat unfair as these aren't specifically trained for web agent tasks

**Presentation Issues:**
- The paper is quite dense and could benefit from more intuitive explanations of the core mechanisms
- Some technical details are relegated to appendices, making the main text occasionally difficult to follow
- The related work section could better position AEPO within the broader landscape of entropy-regularized RL methods

**Significance Assessment:**

The paper makes a meaningful contribution to the growing field of agentic reinforcement learning. The identified entropy-balancing problems are practically relevant for training web agents, and the proposed solutions appear effective. The results demonstrate that careful entropy management can lead to substantial performance improvements in complex, multi-turn reasoning tasks. However, the approach is somewhat specialized to the web agent domain, and its general applicability to other RL settings would need further validation.

Overall, this represents a solid technical contribution with strong empirical results, though some theoretical aspects and broader implications could be more thoroughly developed.

---

# LaSeR: Reinforcement Learning with Last-Token Self-Rewarding

Authors: Wenkai Yang, Weijie Liu, Ruobing Xie, Yiju Guo, Lulu Wu, Saiyong Yang, Yankai Lin

Keywords: Reinforcement Learning, Self-Rewarding, Large Language Models, Reasoning, Self-Verification, Last-Token, RLVR

Comments: Work in progress. Github repo: https://github.com/RUCBM/LaSeR

Paper link: [http://arxiv.org/abs/2510.14943v1](http://arxiv.org/abs/2510.14943v1)

## Abstract

Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as a core paradigm for enhancing the reasoning capabilities of Large Language Models (LLMs). To address the lack of verification signals at test time, prior studies incorporate the training of model's self-verification capability into the standard RLVR process, thereby unifying reasoning and verification capabilities within a single LLM. However, previous practice requires the LLM to sequentially generate solutions and self-verifications using two separate prompt templates, which significantly reduces efficiency. In this work, we theoretically reveal that the closed-form solution to the RL objective of self-verification can be reduced to a remarkably simple form: the true reasoning reward of a solution is equal to its last-token self-rewarding score, which is computed as the difference between the policy model's next-token log-probability assigned to any pre-specified token at the solution's last token and a pre-calculated constant, scaled by the KL coefficient. Based on this insight, we propose LaSeR (Reinforcement Learning with Last-Token Self-Rewarding), an algorithm that simply augments the original RLVR loss with a MSE loss that aligns the last-token self-rewarding scores with verifier-based reasoning rewards, jointly optimizing the reasoning and self-rewarding capabilities of LLMs. The optimized self-rewarding scores can be utilized in both training and testing to enhance model performance. Notably, our algorithm derives these scores from the predicted next-token probability distribution of the last token immediately after generation, incurring only the minimal extra cost of one additional token inference. Experiments show that our method not only improves the model's reasoning performance but also equips it with remarkable self-rewarding capability, thereby boosting its inference-time scaling performance.

## Summary

Here is a summary of the paper "LaSeR: Reinforcement Learning with Last-Token Self-Rewarding":

**Key Contribution:** The paper proposes LaSeR, a lightweight and efficient method that jointly optimizes reasoning and self-verification capabilities in large language models (LLMs) within the Reinforcement Learning with Verifiable Rewards (RLVR) framework. The key innovation is deriving self-rewarding signals directly from the model's last-token probability distribution, eliminating the need for separate verification generation.

**Methodology:** The authors theoretically show that the optimal solution to the RL verification objective can be simplified to a "last-token self-rewarding score" - the difference between the policy model's log-probability for a pre-specified special token at the final response token and a pre-calculated constant, scaled by the KL coefficient. This allows them to replace explicit RL optimization for self-verification with a simple MSE loss that aligns these self-rewarding scores with verifier-based reasoning rewards. The method requires only one additional token inference and can be seamlessly integrated into existing RLVR frameworks like GRPO.

**Results:** Experiments across LLaMA and Qwen architectures on mathematical reasoning benchmarks (MATH500, AMC23, AIME24/25, OlympiadBench) demonstrate that LaSeR not only improves reasoning performance but also achieves remarkable self-verification capability (~80% F1 scores), outperforming equally-sized external verifiers and matching the performance of a 72B reward model. The method also enhances inference-time scaling through weighted majority voting using the self-rewarding scores. Additional experiments show the approach generalizes to general reasoning tasks, though with somewhat reduced effectiveness compared to mathematical reasoning.

The paper presents a highly efficient solution to the self-verification problem in RLVR, enabling models to assess their own outputs with minimal computational overhead while maintaining or improving reasoning performance.

## Critique

Of course. Here is a commentary on the strengths and weaknesses of the paper "LaSeR: Reinforcement Learning with Last-Token Self-Rewarding".

### **Strengths**

1.  **High Novelty and Conceptual Elegance:** The core insight of the paper is highly novel. The theoretical derivation showing that the optimal solution for self-verification can be reduced to a simple function of the *last token's* probability distribution is elegant and surprising. This reframes the complex problem of training a separate verification module into a lightweight, almost cost-free alignment task. The idea of encoding a model's confidence in its own solution into a single, pre-specified token's probability is a clever and unconventional approach.

2.  **Significant Practical Impact and Efficiency:** The paper delivers a method with a compelling practical advantage: near-zero additional computational cost. By requiring only one (or potentially zero) additional token inference, LaSeR is drastically more efficient than prior self-verification methods that require a separate, full-generation verification step. This makes it highly attractive for real-world deployment. The empirical results showing that a model's own self-rewarding score can rival the performance of a separate, large (72B) reward model is a significant and impressive result.

3.  **Strong and Comprehensive Empirical Validation:** The authors provide thorough experimentation across multiple model architectures (LLaMA, Qwen), model states (pre-trained, mid-trained, reinforced), and benchmarks. The consistent improvement in both reasoning accuracy and self-verification F1 score demonstrates the robustness of the method. The inclusion of inference-time scaling results (weighted majority voting) shows that the learned self-rewarding capability has direct, practical utility beyond just being a diagnostic tool.

4.  **Clear and Methodical Presentation:** The paper is well-structured. It effectively builds from the theoretical foundation to the practical algorithm, and then to the experimental validation. The use of a clear algorithm box (Algorithm 1) and an illustrative figure (Figure 1) helps in understanding the proposed method. The discussion of limitations and potential future variants (Section 5.3) adds depth and honesty to the presentation.

### **Weaknesses**

1.  **Limited Exploration of General Reasoning:** While the paper demonstrates stellar performance in mathematical reasoning, its generalizability is shown to be more limited. The results on MMLU-Pro and GPQA (Section 5.2) indicate that the self-rewarding capability does not reach the same high level of accuracy in general domains. The authors' speculation about the reasons (weaker base model capability, noisier verifier) is plausible, but this remains a clear limitation of the current work. It suggests that the method's effectiveness might be somewhat domain-specific or dependent on high-quality verification signals.

2.  **Theoretical Approximation and its Implications:** The entire method rests on a key approximation: that the partition function \( Z(\bm{x}, \bm{y}) \approx 1 \), which allows the simplification in Equation 11. While the justification (that \( \pi_{ref}(z_c|\bm{x},\bm{y}) \) is extremely small) is supported by empirical observation (Figure 11), it remains an approximation. The paper could benefit from a more rigorous theoretical discussion of the error bounds of this approximation and under what conditions it might break down.

3.  **Hyperparameter Sensitivity and Ablation Depth:** The method introduces several new hyperparameters (\( \beta_v, \alpha, \tau \), warm-up steps). While an appendix provides an ablation (Appendix D), a more detailed analysis of the sensitivity of the results to these choices would strengthen the paper. For instance, how critical is the class-level re-weighting? How does the performance change if the pre-specified token \( z_c \) is chosen differently?

4.  **Comparison to Broader Baselines:** The primary baseline is GRPO without self-verification. While this is a logical comparison, it would be even more compelling to see a direct, controlled comparison against a prior self-verification method (e.g., one that generates "Yes"/"No" after the solution) in terms of both final performance and, crucially, computational cost (FLOPs/training time). This would quantitatively highlight LaSeR's efficiency advantage.

### **Overall Assessment**

This is a high-quality paper that presents a novel, efficient, and effective method for joint reasoning and self-verification. Its core theoretical insight is elegant and has significant practical implications. The strength of the results in mathematical reasoning is compelling. The main weaknesses lie in the demonstrated limitations for general reasoning tasks and the reliance on a key theoretical approximation. Despite these, the paper makes a substantial contribution by introducing a highly efficient paradigm for self-rewarding that could become a standard component in the training of future reasoning models.

---

# Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn LLM Agents

Authors: Guoqing Wang, Sunhao Dai, Guangze Ye, Zeyu Gan, Wei Yao, Yong Deng, Xiaofeng Wu, Zhenzhe Ying

Keywords: Multi-agent communication, Reinforcement Learning, Multi-turn LLM Agents, Information Gain, Policy Optimization, Reward Sparsity, Advantage Collapse

Comments: None

Paper link: [http://arxiv.org/abs/2510.14967v1](http://arxiv.org/abs/2510.14967v1)

## Abstract

Large language model (LLM)-based agents are increasingly trained with reinforcement learning (RL) to enhance their ability to interact with external environments through tool use, particularly in search-based settings that require multi-turn reasoning and knowledge acquisition. However, existing approaches typically rely on outcome-based rewards that are only provided at the final answer. This reward sparsity becomes particularly problematic in multi-turn settings, where long trajectories exacerbate two critical issues: (i) advantage collapse, where all rollouts receive identical rewards and provide no useful learning signals, and (ii) lack of fine-grained credit assignment, where dependencies between turns are obscured, especially in long-horizon tasks. In this paper, we propose Information Gain-based Policy Optimization (IGPO), a simple yet effective RL framework that provides dense and intrinsic supervision for multi-turn agent training. IGPO models each interaction turn as an incremental process of acquiring information about the ground truth, and defines turn-level rewards as the marginal increase in the policy's probability of producing the correct answer. Unlike prior process-level reward approaches that depend on external reward models or costly Monte Carlo estimation, IGPO derives intrinsic rewards directly from the model's own belief updates. These intrinsic turn-level rewards are combined with outcome-level supervision to form dense reward trajectories. Extensive experiments on both in-domain and out-of-domain benchmarks demonstrate that IGPO consistently outperforms strong baselines in multi-turn scenarios, achieving higher accuracy and improved sample efficiency.

## Summary

Based on the provided paper, here is a summary focusing on its key contributions, methods, and results:

**Title:** Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn LLM Agents

### Key Contributions
This paper introduces **Information Gain-based Policy Optimization (IGPO)**, a novel reinforcement learning framework designed to address the critical limitations of sparse outcome-based rewards in multi-turn LLM agents. The primary contributions are:
1.  **Problem Identification:** It clearly identifies and analyzes the "advantage collapse" phenomenon in standard outcome-reward-based RL (like GRPO), where long trajectories with identical final answers yield zero learning signals, a problem especially severe for smaller models.
2.  **Intrinsic Reward Design:** It proposes a simple yet effective intrinsic, turn-level reward based on "information gain." This reward measures the marginal increase in the policy's probability of generating the correct (ground-truth) answer after each agent-environment interaction turn.
3.  **Dense Supervision:** By combining these dense, turn-level information gain rewards with the final outcome reward, IGPO provides fine-grained credit assignment throughout the entire multi-turn trajectory, mitigating reward sparsity and guiding the agent more effectively.

### Proposed Method: IGPO
The core of IGPO lies in its reward mechanism:
- **Turn-level Reward Calculation:** For each turn in a multi-step interaction (e.g., with a search engine), the model computes the probability of the ground-truth answer given the trajectory up to that point. The reward for that turn is the *increase* in this probability compared to the previous turn: `r_t = π_θ(a | q, o_≤t) - π_θ(a | q, o_≤t-1)`.
- **Integrated Reward Signal:** The final reward trajectory for a rollout is a sequence of these information gain rewards for all intermediate turns, capped with the standard outcome-based F1 score (or a format penalty) for the final answer turn.
- **Policy Optimization:** These per-turn rewards are normalized within a group of sampled rollouts, discounted, and summed to compute turn-level advantages. The policy is then optimized using a GRPO-style objective function but applied at the turn level with these new advantages, providing a much denser and more informative learning signal.

### Key Results
Extensive experiments on search-based agent tasks across seven in-domain (NQ, TQ, HotpotQA, 2Wiki) and out-of-domain (MusiQue, Bamboogle, PopQA) benchmarks demonstrate IGPO's effectiveness:
- **State-of-the-Art Performance:** IGPO outperforms a wide range of strong baselines, including prompt-based methods, outcome-reward RL methods (Search-r1, DeepResearcher), and other step-reward RL methods (StepSearch, ReasoningRAG, GiGPO). It achieves a notable average score of **58.7**, a +4.8 point improvement over the best baseline.
- **Superior Sample Efficiency:** IGPO demonstrates faster convergence and higher performance per token used in training compared to GRPO, indicating significantly improved token efficiency.
- **Effectiveness for Smaller Models:** The performance gains are especially pronounced for smaller models (e.g., Qwen2.5-3B), where advantage collapse is more severe. IGPO improved the 3B model by +15.3 points over standard GRPO.
- **Ablation Studies:** Ablations confirm that both the information gain reward and the final outcome reward are crucial and complementary. Using only the information gain reward already performs comparably to or better than GRPO, demonstrating its robustness and intrinsic value.

In conclusion, IGPO provides a simple, intrinsic, and highly effective solution to the reward sparsity problem in multi-turn LLM agents, leading to substantial improvements in performance, stability, and sample efficiency.

## Critique

Of course. Here is a critique of the paper "Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn LLM Agents."

### Strengths

1.  **Novel and Intuitive Core Idea:** The central concept of using the marginal increase in the model's own probability of generating the correct answer as an intrinsic, turn-level reward is both novel and elegant. It directly addresses the problem of "advantage collapse" in a theoretically grounded way without introducing complex external components like a separate reward model or requiring expensive Monte Carlo rollouts.
2.  **Significant and Well-Validated Results:** The empirical results are a major strength. The paper demonstrates clear and consistent state-of-the-art performance across a wide range of in-domain and out-of-domain benchmarks. The ablation studies are particularly compelling, showing that:
    *   The combination of Information Gain (IG) and outcome reward is crucial for peak performance.
    *   IGPO provides a substantial boost, especially for smaller models (e.g., +15.3 points on a 3B model), which powerfully validates its claim of mitigating reward sparsity.
    *   It achieves higher "token efficiency," meaning it learns more effectively from the same amount of data.
3.  **Clear Problem Formulation:** The paper does an excellent job of framing the problem. The issues of "advantage collapse" and the lack of "fine-grained credit assignment" in multi-turn, outcome-reward-only RL are clearly explained and empirically supported (e.g., with Figure 1). This makes the motivation for the work very strong.
4.  **Comprehensive Evaluation:** The comparison is thorough, including prompt-based baselines, outcome-reward RL methods, other step-reward RL methods, and a suite of standard RL algorithms (PPO, GRPO, etc.). This leaves little doubt about IGPO's relative performance.
5.  **Clarity and Reproducibility:** The methodology is described with sufficient detail, including the reward formulation, advantage calculation, and policy optimization objective. The inclusion of a GitHub repository and appendices with implementation details and prompts is a significant plus for reproducibility.

### Weaknesses

1.  **Key Limitation: Dependency on Ground Truth:** The most significant limitation, which the authors explicitly acknowledge, is the method's reliance on a ground-truth answer to compute the information gain. This restricts its application to tasks where such answers are known and static (e.g., QA on existing benchmarks). It cannot be directly applied to open-ended, creative, or subjective tasks where there is no single "correct" answer. This is a substantial constraint on the generalizability of the approach.
2.  **Computational Overhead:** While the method avoids the cost of training a reward model or running MCTS, it does introduce a non-trivial computational cost. For every turn in every rollout, the model must perform a forward pass on the entire ground-truth answer sequence under teacher forcing (Equation 3). For long answers and many rollouts, this could significantly increase training time and memory usage compared to outcome-only methods.
3.  **Theoretical Analysis Could Be Deeper:** The theoretical analysis provided in the appendix, while a positive addition, is relatively brief. A more rigorous analysis connecting the proposed reward to established information-theoretic principles or providing stronger guarantees on convergence/performance would strengthen the paper's theoretical contribution.
4.  **Limited Exploration of the Discount Factor (γ):** The paper sets the discount factor γ to 1.0 and does not explore its tuning. While this simplifies the method, it is a non-standard choice in RL. An ablation on γ would have been informative to show whether a more traditional discounted return provides any benefit or if γ=1 is indeed optimal for this specific reward structure.

### Overall Assessment

This is a **strong and impactful paper**. It identifies a clear and important problem (reward sparsity in multi-turn agents) and proposes a simple, intuitive, and highly effective solution. The novelty of the "information gain" reward is high, and the empirical results are robust and convincing, clearly establishing a new state-of-the-art for the tasks evaluated.

The primary weaknesses are the inherent limitation of requiring ground-truth answers and the associated computational cost, which are important considerations for future applications. However, within its intended domain of supervised, goal-oriented multi-turn tasks, the approach represents a significant advance. The presentation is clear, and the work is likely to influence subsequent research in agent training and reinforcement learning for language models.

---

# Beyond Correctness: Evaluating Subjective Writing Preferences Across Cultures

Authors: Shuangshuang Ying, Yunwen Li, Xingwei Qu, Xin Li, Sheng Jin, Minghao Liu, Zhoufutu Wen, Xeron Du, Tianyu Zheng, Yichi Zhang, Letian Ni, Yuyang Cheng, Qiguang Chen, Jingzhe Ding, Shengda Long, Wangchunshu Zhou, Jiazhan Feng, Wanjun Zhong, Libo Qin, Ge Zhang, Wenhao Huang, Wanxiang Che, Chenghua Lin

Keywords: Subjective Writing Evaluation, Human Preference Modeling, Reward Models, Creative Writing Assessment, Cross-Lingual Benchmarking, Generative Reward Architectures

Comments: None

Paper link: [http://arxiv.org/abs/2510.14616v1](http://arxiv.org/abs/2510.14616v1)

## Abstract

Current preference learning methods achieve high accuracy on standard benchmarks but exhibit significant performance degradation when objective quality signals are removed. We introduce WritingPreferenceBench, a dataset of 1,800 human-annotated preference pairs (1,200 English, 600 Chinese) across 8 creative writing genres, where responses are matched for objective correctness, factual accuracy, and length. On this benchmark, sequence-based reward models--the standard architecture for RLHF--achieve only 52.7% mean accuracy, while zero-shot language model judges perform at 53.9%. In contrast, generative reward models that produce explicit reasoning chains achieve 81.8% accuracy. We observe high within-model variance across genres: individual models range from 18.2% to 81.8% accuracy across different writing categories, with standard deviations averaging 10.1%. This variance persists regardless of model scale, with 27B parameter models showing no consistent improvement over 8B variants. Our results suggest that current RLHF methods primarily learn to detect objective errors rather than capture subjective quality preferences (e.g., creativity, stylistic flair, and emotional resonance), and that successful preference modeling may require intermediate reasoning representations rather than direct classification.

## Summary

Of course. Here is a summary of the paper "Beyond Correctness: Evaluating Subjective Writing Preferences Across Cultures":

This paper introduces **WritingPreferenceBench**, a novel benchmark designed to evaluate how well AI models can understand and predict *subjective* human preferences in creative writing, moving beyond objective measures like grammatical correctness or factual accuracy. The core contribution is a meticulously constructed dataset of **1,800 human-annotated preference pairs** (1,200 in English, 600 in Chinese) across 8 creative genres (e.g., poetry, fiction, scriptwriting). The key innovation is that in each pair, the two responses are carefully matched for objective quality (grammar, factuality, length), forcing models to judge based on purely subjective qualities like creativity, stylistic flair, and emotional resonance.

The paper's main methodological contribution lies in its rigorous, multi-stage data curation pipeline. It involves generating diverse responses from 20 state-of-the-art models and then applying a strict human-in-the-loop annotation protocol with trained experts. This process ensures the final preference pairs reflect genuine aesthetic distinctions rather than confounding variables.

The results reveal critical limitations in current preference learning paradigms. The authors evaluate 21 models, including standard sequence-based reward models (the dominant architecture in RLHF systems) and generative reward models that produce reasoning chains. The key findings are:
*   **Sequence-based reward models fail catastrophically**, achieving only **52.7% accuracy**—barely above random chance. This indicates they primarily learn to detect objective errors rather than nuanced aesthetic preferences.
*   **Generative reward models with explicit reasoning** significantly outperform others, with the best model (**RM-R1-Qwen2.5-7B**) achieving **81.8% accuracy**. This suggests that structured intermediate reasoning is crucial for subjective judgment.
*   **All models exhibit severe instability** across different writing genres, with individual model performance swinging wildly (e.g., from 18.2% to 81.8%). This indicates they rely on brittle, genre-specific heuristics rather than learning generalizable principles of quality.
*   **General-purpose LLMs used as zero-shot judges** also perform poorly (mean **53.9% accuracy**), and advanced reasoning capabilities (e.g., Chain-of-Thought) do not systematically improve their performance.

In conclusion, the paper demonstrates that current RLHF methods are fundamentally misaligned with the demands of subjective creative tasks. Its benchmark and findings highlight the need for new architectures and training objectives that can capture nuanced human aesthetic preferences, with explicit reasoning emerging as a promising direction.

## Critique

Based on the provided paper "Beyond Correctness: Evaluating Subjective Writing Preferences Across Cultures," here is an analysis of its strengths and weaknesses:

### Strengths

**Novelty and Research Gap:**
- The paper addresses a clear and important gap in current preference learning benchmarks by focusing specifically on *subjective* writing quality, isolating it from objective correctness (grammar, factuality, length). This is a novel and valuable contribution, as most existing benchmarks (e.g., RewardBench) conflate objective and subjective quality signals.
- The introduction of a **cross-lingual dataset (English and Chinese)** for subjective preference evaluation is significant, as it extends the analysis beyond the typical English-centric focus of most NLP research.

**Rigor in Dataset Construction:**
- The **meticulous, multi-stage data curation pipeline** involving expert annotators, strict filtering for objective confounds, and human validation is a major strength. The use of a detailed rubric and inter-annotator agreement measures adds credibility.
- The **statistical validation** of the dataset (e.g., length distributions, score gaps) convincingly demonstrates that it successfully captures genuine subjective preference distinctions.

**Significance of Results:**
- The key finding—that **sequence-based reward models (the standard in RLHF) perform near-randomly (~53%) on purely subjective tasks**—is striking and has important implications for the field. It suggests that current RLHF methods may be primarily optimizing for error detection rather than genuine aesthetic alignment.
- The **strong performance of generative reward models with explicit reasoning chains (~82% accuracy)** provides a clear architectural insight and a promising direction for future work.
- The **high variance across genres** for all models exposes a critical brittleness in current preference learning systems, highlighting that they rely on superficial, genre-specific heuristics rather than generalizable principles of quality.

**Clarity and Structure:**
- The paper is **well-structured** with clear sections, and the use of tables and figures effectively summarizes key results (e.g., model performance across genres).
- The **contributions are clearly stated** at the end of the introduction, and the discussion section does a good job of synthesizing the implications of the empirical findings.

### Weaknesses

**Limited Model Scale and Diversity:**
- While 21 models were evaluated, the scale of the largest models (e.g., 27B parameters) is modest compared to today's state-of-the-art (e.g., models with 70B+ parameters or proprietary giants like GPT-4). The conclusion that "scale does not help" might not hold for significantly larger models.
- The set of **generative reward models is relatively small (only 3 from the RM-R1 series)**, making it difficult to generalize the finding that generative architectures are superior. More diversity in generative reward model designs would strengthen this claim.

**Narrow Focus on Writing:**
- The benchmark is exclusively focused on **creative writing genres**. While this is a valid and important domain, the conclusions about subjective preference modeling might not generalize to other subjective tasks (e.g., dialogue, humor, art critique). A broader range of subjective tasks could make the findings more robust.

**Lack of Detailed Error Analysis:**
- The paper identifies *that* models fail catastrophically on certain genres (e.g., Poetry, Scriptwriting) but provides limited analysis into *why*. A deeper qualitative analysis of model failures (e.g., what heuristics they are relying on, what aspects of "creativity" or "emotional resonance" they miss) would be highly valuable.

**Clarity on "Reasoning":**
- The paper attributes the success of generative RMs to "explicit reasoning chains," but it does not deeply analyze the *content* or *quality* of these reasoning steps. It is unclear if the models are performing genuine reasoning or simply leveraging a more flexible architecture that happens to work better. A deeper dive into the reasoning chains would strengthen this argument.

**Presentation Minor Issues:**
- The use of colored cells in the results tables (e.g., red for <50%, green for >70%) is helpful, but in a plain text/markdown format without actual color, this information is lost. The captions should explicitly state the meaning of the coloring for accessibility.
- Some acronyms are used before being fully defined (e.g., RLHF is used in the abstract before its full form appears in the introduction).

### Overall Assessment

This is a **strong, well-executed paper** that makes a valuable contribution by rigorously demonstrating a critical limitation in current preference learning paradigms. The novel benchmark, clear empirical results, and important architectural insights significantly advance our understanding of subjective alignment in language models. The main weaknesses relate to the scope of models and tasks evaluated and a somewhat superficial treatment of the "reasoning" mechanism in generative RMs, but these do not undermine the core contributions. The work has clear implications for the future development of RLHF and reward modeling techniques.

---

# Retrofitting Small Multilingual Models for Retrieval: Matching 7B Performance with 300M Parameters

Authors: Lifu Tu, Yingbo Zhou, Semih Yavuz

Keywords: Multilingual Embedding Models, Information Retrieval, Synthetic Data Generation, Model Efficiency, Contrastive Learning, Hard Negatives, Task Diversity, Language Models

Comments: None

Paper link: [http://arxiv.org/abs/2510.14274v1](http://arxiv.org/abs/2510.14274v1)

## Abstract

Training effective multilingual embedding models presents unique challenges due to the diversity of languages and task objectives. Although small multilingual models (<1 B parameters) perform well on multilingual tasks generally, they consistently lag behind larger models (>1 B) in the most prevalent use case: retrieval. This raises a critical question: Can smaller models be retrofitted specifically for retrieval tasks to enhance their performance? In this work, we investigate key factors that influence the effectiveness of multilingual embeddings, focusing on training data scale, negative sampling strategies, and data diversity. We find that while increasing the scale of training data yields initial performance gains, these improvements quickly plateau - indicating diminishing returns. Incorporating hard negatives proves essential for consistently improving retrieval accuracy. Furthermore, our analysis reveals that task diversity in the training data contributes more significantly to performance than language diversity alone. As a result, we develop a compact (approximately 300M) multilingual model that achieves retrieval performance comparable to or even surpassing current strong 7B models.

## Summary

This paper presents a method for retrofitting small multilingual embedding models (≈300M parameters) to achieve retrieval performance comparable to or even surpassing larger 7B-parameter models. The key contributions include identifying critical factors that influence multilingual embedding effectiveness and developing a compact model that achieves state-of-the-art results on multilingual retrieval tasks.

The method involves strategically fine-tuning small multilingual models using a combination of high-quality English retrieval data and synthetic multilingual query-document pairs generated from mC4 Wikipedia articles using GPT-4o-mini. The authors systematically investigate several key factors: (1) data sources - showing that synthetic multilingual data significantly improves performance while parallel data contributes little; (2) data scale - finding diminishing returns beyond 4k examples per language; (3) hard negative mining - demonstrating consistent improvements when using negatives mined with a strong backbone model; and (4) diversity analysis - revealing that task diversity contributes more to performance than language diversity alone.

Results show the proposed 300M-parameter model achieves a score of 60.56 on MMTEB multilingual retrieval tasks, outperforming or matching current strong 7B models like SFR-Embedding-Mistral (59.44) and gte-Qwen2-7B-instruct (60.08). The model also shows consistent improvements across other task categories, with an overall MMTEB score 1.15 points higher than the baseline. This work demonstrates that small models can be effectively optimized for retrieval, the most critical application of embedding models, through careful data curation and training strategies.

## Critique

Of course. Here is a critique of the paper "Retrofitting Small Multilingual Models for Retrieval: Matching 7B Performance with 300M Parameters."

### Strengths

1.  **High Practical Significance and Clear Contribution:** The paper addresses a critical and highly practical problem: the performance gap between small and large models in retrieval, which is the core of RAG systems. The central claim—achieving 7B-level performance with a 300M parameter model—is compelling and, if widely applicable, has significant implications for the efficiency and deployment cost of real-world AI systems.

2.  **Rigorous and Systematic Analysis:** The paper's major strength lies in its thorough experimental analysis. It doesn't just present a new model; it systematically deconstructs the factors contributing to its success:
    *   **Data Source:** It validates the utility of synthetic data over parallel data for retrieval.
    *   **Data Scale:** It identifies the point of diminishing returns (plateau after 4k samples per language), which is a valuable finding for resource-constrained training.
    *   **Hard Negatives:** It confirms the importance of this established but crucial technique.
    *   **Task vs. Language Diversity:** The most insightful finding is that **task diversity contributes more to performance than language diversity** for retrofitting. This challenges a potential assumption that merely adding more languages is the key and shifts the focus to the variety of tasks during training.

3.  **Strong Empirical Results:** The results are impressive. Outperforming several 7B models on the MMTEB retrieval benchmark (60.56 vs. 59.44 for SFR-Embedding-Mistral) with a model ~23x smaller is a concrete and substantial achievement. The improved performance across other task categories (Table 4) further demonstrates the general robustness of the approach.

4.  **Clarity and Structure:** The paper is well-structured and clearly written. The progression from introduction to analysis to the final model is logical. The tables and figures are effective in supporting the narrative.

### Weaknesses

1.  **Limited Novelty in Core Techniques:** The methodology itself is an expert application and combination of existing, well-established techniques rather than a novel algorithmic contribution. The use of contrastive loss, synthetic data generation with LLMs, and hard negative mining are all standard practices in modern embedding training. The paper's primary novelty lies in the **systematic investigation and demonstration** of how these techniques interact to enable small-model retrofitting, rather than in the techniques themselves.

2.  **Unexplored Aspects of Synthetic Data:** While the paper shows synthetic data is effective, it lacks a deeper analysis of its qualities.
    *   **Quality vs. Quantity:** The analysis focuses on scale but not on the quality of the generated queries. How does the quality of GPT-4o-mini generated data compare to human-annotated data or data from a more powerful model?
    *   **Diversity of Queries:** The prompts used for generation (Appendix A.2) are simple. A more diverse set of instructions or query types could potentially lead to better performance and is an unexplored variable.

3.  **Superficial Treatment of Limitations:** The "Limitations" section, while acknowledging key issues, is somewhat brief.
    *   **Benchmark Bias:** The heavy reliance on MMTEB, which has limited retrieval tasks for low-resource languages, is a valid limitation. However, the paper doesn't provide any analysis or results on a curated set of truly low-resource languages to test the method's limits.
    *   **Computational Cost of Data Generation:** The cost and carbon footprint of generating synthetic data for 20 languages using a powerful model like GPT-4o-mini is not discussed, which is a non-trivial aspect of the proposed pipeline.

4.  **Clarity Gaps in the Final Model Recipe:** The description of the final, best-performing model ("Our" in Table 1) is somewhat implicit. It is pieced together from the analysis sections but could be more explicitly summarized. For instance, it would be helpful to have a clear, consolidated statement: "Our final model was trained on [En-Mix] + [Mul-Syn] with hard negatives, using [X] total samples."

### Overall Assessment

This is a **high-quality, engineering-focused paper** with significant practical value. Its primary contribution is not a new algorithm but a comprehensive blueprint and empirical proof that small multilingual models can be retrofitted for state-of-the-art retrieval performance through a careful, data-centric strategy. The analysis revealing the greater importance of task diversity over language diversity is a key intellectual takeaway. While the core techniques are not novel, the successful integration and rigorous evaluation of these methods to solve a pressing problem constitute a solid contribution to the field. The paper is clear, well-supported by evidence, and its findings are likely to influence how efficient multilingual embedding models are developed.

---

# LiRA: Linguistic Robust Anchoring for Cross-lingual Large Language Models

Authors: Haolin Li, Haipeng Zhang, Mang Li, Yaohua Wang, Lijie Wen, Yu Zhang, Biqing Huang

Keywords: Cross-lingual Large Language Models, Multilingual Representation Anchoring, Low-resource Language Adaptation, Cross-lingual Retrieval, Multilingual Reasoning, Representation Stability, Linguistic Robustness

Comments: None

Paper link: [http://arxiv.org/abs/2510.14466v1](http://arxiv.org/abs/2510.14466v1)

## Abstract

As large language models (LLMs) rapidly advance, performance on high-resource languages (e.g., English, Chinese) is nearing saturation, yet remains substantially lower for low-resource languages (e.g., Urdu, Thai) due to limited training data, machine-translation noise, and unstable cross-lingual alignment. We introduce LiRA (Linguistic Robust Anchoring for Large Language Models), a training framework that robustly improves cross-lingual representations under low-resource conditions while jointly strengthening retrieval and reasoning. LiRA comprises two modules: (i) Arca (Anchored Representation Composition Architecture), which anchors low-resource languages to an English semantic space via anchor-based alignment and multi-agent collaborative encoding, preserving geometric stability in a shared embedding space; and (ii) LaSR (Language-coupled Semantic Reasoner), which adds a language-aware lightweight reasoning head with consistency regularization on top of Arca's multilingual representations, unifying the training objective to enhance cross-lingual understanding, retrieval, and reasoning robustness. We further construct and release a multilingual product retrieval dataset covering five Southeast Asian and two South Asian languages. Experiments across low-resource benchmarks (cross-lingual retrieval, semantic similarity, and reasoning) show consistent gains and robustness under few-shot and noise-amplified settings; ablations validate the contribution of both Arca and LaSR. Code will be released on GitHub and the dataset on Hugging Face.

## Summary

Of course. Here is a summary of the paper "LiRA: Linguistic Robust Anchoring for Cross-lingual Large Language Models".

### Summary

**LiRA** is a novel training framework designed to improve the performance of Large Language Models (LLMs) on low-resource languages (LRLs), which typically lag behind high-resource languages like English due to limited training data and unstable cross-lingual alignment. The key idea is to "anchor" the representations of LRLs to the robust English semantic space, thereby transferring the strong reasoning and retrieval capabilities of English-centric LLMs.

The framework consists of two main components:
1.  **Arca (Anchored Representation Composition Architecture):** This module explicitly aligns LRL representations with English through a multi-agent system. It uses an "Embedding Critic" to minimize the distance between a direct LRL embedding and its English translation, and a "Translation Critic" (an LLM judge) to assess the semantic, emotional, and pragmatic fidelity of translations. An "Actor" then selects the best translation candidate based on these scores, jointly reducing semantic drift and translation noise.
2.  **LaSR (Language-coupled Semantic Reasoner):** This is a lightweight reasoning head that fuses the multilingual and English embeddings from Arca. It employs queue-based training objectives (a correlation queue for ranking and a document queue for retrieval) to enhance cross-lingual understanding and reasoning robustness in a unified manner.

A significant contribution is the release of **LazRetrieval**, a new multilingual product retrieval dataset covering seven Southeast and South Asian low-resource languages (Vietnamese, Thai, Indonesian, Malay, Urdu, Bengali, and Filipino), facilitating further research in this area.

### Key Results
Experiments across retrieval, sentence ranking, and reasoning tasks demonstrate LiRA's effectiveness:
*   **Retrieval & Ranking:** LiRA consistently outperforms the strong Qwen3-Embedding baseline and previous state-of-the-art methods like LUSIFER on public benchmarks (MLQARetrieval, BelebeleRetrieval, STS22) and the new LazRetrieval dataset, with particular gains on low-resource languages.
*   **Reasoning:** On mathematical reasoning (MGSM) and reading comprehension (X-CSQA) tasks, LiRA achieves small but consistent improvements over the powerful Qwen3-14B model, especially on typologically distant or lower-resource languages, without harming English performance.
*   **Ablation Studies:** Confirm that all components of LiRA (the LLM Critic, Embedding Critic, translation module, and FIFO queues) are essential, and their removal leads to significant performance drops, validating the design's synergy.

## Critique

Of course. Here is a balanced assessment of the strengths and weaknesses of the paper "LiRA: Linguistic Robust Anchoring for Cross-lingual Large Language Models."

### Strengths

1.  **Strong Theoretical Foundation:** This is the paper's most significant strength and a key differentiator from prior work. The authors don't just present a new architecture; they ground it in a formal framework with assumptions, definitions, a theorem, and a corollary. This provides a rigorous justification for their design choices (like the concatenation of representation paths) and offers measurable error bounds, which is rare in this domain.
2.  **Comprehensive and Unified Approach:** The proposed LiRA framework is not a single trick but a complete system with two well-integrated components:
    *   **Arca:** A sophisticated multi-agent (critic-actor) system that explicitly minimizes both anchoring error and translation distortion.
    *   **LaSR:** A practical module for fusing representations and training with queue-based objectives for robust retrieval and reasoning.
    This holistic approach allows the model to excel across multiple task types (retrieval, ranking, reasoning) within a single framework.
3.  **Significant Empirical Results:** The paper demonstrates consistent, albeit sometimes modest, improvements over a very strong baseline (Qwen3). Crucially, the gains are most pronounced on low-resource languages, which is the core problem the paper aims to solve. The comprehensive evaluation across multiple benchmarks (MLQA, Belebele, STS22, MGSM, X-CSQA) adds credibility.
4.  **Valuable Resource Contribution:** The release of the **LazRetrieval** dataset, covering seven Southeast and South Asian languages, is a substantial contribution to the community. It addresses the critical need for benchmarks in truly low-resource settings and will facilitate future research.
5.  **Clear Ablation Study:** The ablation in Table 5 is effective and clear. It successfully demonstrates the importance of each core component, showing that the LLM Critic, Embedding Critic, and FIFO queue are all essential to the system's performance.

### Weaknesses

1.  **Marginal Gains on Strong Baselines:** While the improvements are consistent, they are often quite small (e.g., +0.1 on MGSM average, +0.5 on X-CSQA average). The paper would be stronger if it included a deeper analysis or statistical significance testing to convince the reader that these gains are meaningful and not just marginal fluctuations, especially when compared to the already high performance of the Qwen3 baseline.
2.  **Complexity and Computational Cost:** The proposed method is architecturally complex, involving multiple encoders, critics, an actor, adaptors, and FIFO queues. The paper does not provide an analysis of the computational overhead, training time, or inference latency compared to the baseline. This complexity could be a barrier to adoption for many researchers and practitioners.
3.  **Clarity of Presentation (Mixed):**
    *   **Technical Sections:** The theoretical section (Section 3) and the detailed method description (Section 4) are dense and challenging to follow for a reader not deeply versed in RKHS embeddings and reinforcement learning. The acronym "Arca" is introduced without a clear expansion until later in the text, which is confusing initially.
    *   **High-Level Flow:** Despite the technical density, the overall narrative is clear: identify the problem, propose a theoretically-grounded solution, break it down into modules, and validate it empirically. The figures (1, 2, 3) are helpful for understanding the architecture.
4.  **Limited Discussion of Limitations:** The conclusion is quite positive and does not dedicate space to a formal discussion of the method's limitations. For instance, the performance on some high-resource languages slightly decreased in the MGSM results; this isn't discussed. Furthermore, the reliance on a high-quality translator `T(x)` is a potential bottleneck that isn't critically examined.

### Overall Assessment

**Novelty:** High. The integration of a rigorous theoretical framework with a multi-agent, critic-based architecture for cross-lingual alignment is a novel and ambitious contribution.

**Significance:** High. The paper tackles a critical problem (LLM performance on low-resource languages) with a comprehensive solution and provides a valuable new dataset. The theoretical grounding elevates it above many purely empirical approaches.

**Clarity:** Good, but could be improved. The high-level story is well-told, but the technical details in Sections 3 and 4 are very dense and could benefit from more intuitive explanations to accompany the formalisms.

In summary, this is a strong, research-heavy paper that makes significant contributions on both theoretical and practical fronts. Its main weaknesses lie in the sometimes-marginal empirical gains over a powerful baseline and the inherent complexity of the proposed system.

