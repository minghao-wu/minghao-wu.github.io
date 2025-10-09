---
title: "ArXiv Daily Digest on 2025-10-08"
collection: digests
permalink: /digests/arxiv_cs_CL_2025-10-08_report
date: 2025-10-08
location: "Online"
---

Today's research landscape showcases significant advances in multi-agent systems, with a clear trend toward sophisticated frameworks that enhance reasoning capabilities through structured collaboration. Several papers introduce innovative multi-agent architectures employing evolutionary search and reinforcement learning (RL) to tackle complex tasks—from automated unit test generation and scientific research automation to multilingual reasoning. Notably, a paradigm shift toward "Markovian Thinking" is emerging, which decouples thinking length from computational context to achieve linear scaling, while other works leverage internal "self signals" like token logits and attention patterns to optimize multi-agent debate efficiency. These approaches collectively highlight a move beyond simple chain-of-thought methods toward more computationally efficient and stateful inference-time strategies, demonstrating substantial performance gains in domains ranging from code generation to mathematical problem-solving.

## TL;DR

**TL;DR: Multi-Agent & Stateful Reasoning for Complex Tasks**  
Recent arXiv papers highlight a shift toward multi-agent collaboration and stateful inference to tackle complex reasoning, code generation, and multilingual tasks. Key themes include:

1. **Multi-Agent Frameworks**:  
   - **Stateful Evolutionary Search** (https://arxiv.org/abs/2510.07147) uses persistent state and adversarial mutation for robust unit-test generation, outperforming stateless baselines.  
   - **Double-Loop Collaboration** (https://arxiv.org/abs/2510.06761) separates plan evolution (professor agents) from execution (student agents) to automate end-to-end scientific research.  
   - **Self-Signal-Driven Debate** (https://arxiv.org/abs/2510.06843) leverages token logits and attention for efficient multi-LLM debates, reducing token use by 40%.  

2. **Efficient Long-Context Reasoning**:  
   - **Markovian Thinking** (https://arxiv.org/abs/2510.06557) chunks reasoning into fixed-size blocks with linear compute, enabling 24K+ token reasoning without quadratic costs.  

3. **Multilingual Reasoning**:  
   - **M-Thinker** (https://arxiv.org/abs/2510.07300) enforces language consistency and cross-lingual alignment via RL, achieving near-perfect consistency and improved accuracy on non-English tasks.  

**Insight**: Combining stateful environments, multi-agent roles, and self-supervision unlocks scalable, efficient reasoning for code, science, and multilingual applications.

---

# A Multi-Agent Framework for Stateful Inference-Time Search

Authors: Arshika Lalan, Rajat Ghosh, Aditya Kolsur, Debojyoti Dutta

Keywords: Multi-Agent Framework, Inference-Time Search, Stateful Reasoning, Evolutionary Algorithms, Unit Test Generation, Code Coverage, Mutation Testing, Adversarial Guidance

Comments: None

Paper link: [http://arxiv.org/abs/2510.07147v1](http://arxiv.org/abs/2510.07147v1)

## Abstract

Recent work explores agentic inference-time techniques to perform structured, multi-step reasoning. However, stateless inference often struggles on multi-step tasks due to the absence of persistent state. Moreover, task-specific fine-tuning or instruction-tuning often achieve surface-level code generation but remain brittle on tasks requiring deeper reasoning and long-horizon dependencies. To address these limitations, we propose stateful multi-agent evolutionary search, a training-free framework that departs from prior stateless approaches by combining (i) persistent inference-time state, (ii) adversarial mutation, and (iii) evolutionary preservation. We demonstrate its effectiveness in automated unit test generation through the generation of edge cases. We generate robust edge cases using an evolutionary search process, where specialized agents sequentially propose, mutate, and score candidates. A controller maintains persistent state across generations, while evolutionary preservation ensures diversity and exploration across all possible cases. This yields a generalist agent capable of discovering robust, high-coverage edge cases across unseen codebases. Experiments show our stateful multi-agent inference framework achieves substantial gains in coverage over stateless single-step baselines, evaluated on prevalent unit-testing benchmarks such as HumanEval and TestGenEvalMini and using three diverse LLM families - Llama, Gemma, and GPT. These results indicate that combining persistent inference-time state with evolutionary search materially improves unit-test generation.

## Summary

This paper introduces a stateful multi-agent evolutionary framework for automated unit test generation, addressing limitations of stateless inference approaches in complex reasoning tasks. The key contribution is a training-free system that maintains persistent state across inference iterations, combining evolutionary search with adversarial guidance to generate robust edge cases with high code coverage.

The methodology employs four specialized agents orchestrated by a controller: an Actor proposes edge cases, an Adversary generates code mutants to test robustness, a Critic integrates coverage metrics and mutation scores into reward signals, and an Executor provides sandboxed evaluation. The framework features a non-Markovian state that preserves edge cases, coverage scores, mutation feedback, and reward history across evolutionary stages. A notable innovation is the cold-start initialization using rule-based heuristics, which often produces high-quality results without iterative refinement in simpler cases.

Experiments on HumanEval and TestGenEvalMini benchmarks using Llama-70B, GPT-o4-mini, and Gemma-2-27B demonstrate that the proposed approach consistently outperforms few-shot and chain-of-thought baselines in line and function coverage. While HumanEval problems were often resolved in single iterations (62% of cases), TestGenEvalMini required extended search, highlighting the framework's ability to scale to complex real-world codebases. The system achieved substantial coverage improvements, though branch coverage showed some variability across models, suggesting opportunities for future refinement of path exploration strategies.

The results validate that stateful inference-time search enables deeper reasoning without model fine-tuning, though at increased computational cost. The work contributes both a novel multi-agent architecture and curated datasets with execution traces, advancing training-free agentic reasoning for software testing applications.

## Critique

Of course. Here is a critique of the paper "A Multi-Agent Framework for Stateful Inference-Time Search," focusing on its strengths and weaknesses.

### Strengths

1.  **Novel and Well-Motivated Core Idea:** The paper's central thesis—that stateless inference is a fundamental limitation for complex, multi-step reasoning tasks—is compelling and timely. The proposal of a **stateful, multi-agent, evolutionary search framework** that operates entirely at inference-time (without fine-tuning) is a significant and novel contribution. It effectively bridges ideas from evolutionary algorithms, multi-agent systems, and adversarial testing.

2.  **Comprehensive and Rigorous Methodology:** The framework is meticulously designed and clearly defined. The breakdown into distinct agents (Actor, Adversary, Critic, Executor, Controller) with precise mathematical formulations (e.g., for State and Reward) is a major strength. The integration of **mutation testing** into the reward function is a particularly clever way to ground the search and prevent overfitting to trivial coverage, moving beyond simple syntactic correctness.

3.  **Strong and Convincing Experimental Design:** The evaluation is thorough. Using two distinct benchmarks—HumanEval (for sanity checking) and the more complex, real-world TestGenEvalMini (for demonstrating the method's value)—was a smart choice. Testing across three different LLM families (Llama, GPT, Gemma) strengthens the claim of the framework's generalizability. The comparison against a suite of strong baselines (zero/one/three-shot with and without CoT) provides a robust performance baseline.

4.  **Honest and Insightful Analysis of Results:** The paper does an excellent job of interpreting its results, including the limitations. The observation that the "cold-start" often solves HumanEval problems is a powerful testament to the strength of the initial heuristics and a honest assessment of benchmark limitations. The nuanced discussion of why branch coverage lags for some models (attributed to a potential bias towards exception-heavy tests) shows a deep understanding of the system's behavior and opens clear avenues for future work.

### Weaknesses

1.  **Significant Computational Cost:** The primary weakness of the approach is its high computational cost, which the authors explicitly acknowledge. The framework requires multiple LLM calls per iteration, mutation analysis, and execution in a sandboxed environment. The runtime graphs for TestGenEvalMini show a steep increase, which could be prohibitive for many practical applications. A more detailed analysis of FLOPs or a direct comparison of total API cost/inference time against the baselines would have made this trade-off even clearer.

2.  **Clarity of Presentation Could Be Improved:**
    *   **Figure 1:** The central architectural diagram is somewhat cluttered and difficult to parse. The relationship between the "N stages" and the flow of data between agents could be visualized more clearly.
    *   **Mathematical Notation:** While the formalism is a strength, the use of both `S_n` and `S_{n-1}` in adjacent definitions, and the slightly complex reward function, can be challenging to follow on a first read. A more narrative walk-through of a single iteration with a concrete example could have improved accessibility.
    *   **Appendix Dependence:** Key details about the "cold-start" and the executor are relegated to the appendix. Given that the cold-start is responsible for a majority of HumanEval solutions, a brief summary of its key heuristics in the main text would be beneficial.

3.  **Limited Discussion of Broader Applicability:** The paper convincingly demonstrates the framework's value for unit test generation. However, the introduction and motivation suggest a much broader potential for "multi-stage reasoning" tasks like theorem proving and program synthesis. A discussion or even a small-scale experiment on another task (e.g., a mathematical reasoning benchmark) would have significantly strengthened the claim of the framework's general utility beyond software testing.

### Overall Assessment

This is a **high-quality, technically sound paper with a novel and impactful contribution**. It identifies a genuine limitation in current LLM inference paradigms and proposes a sophisticated, well-engineered solution. The strengths in novelty, methodological rigor, and experimental validation far outweigh the weaknesses.

The main trade-off is between performance and computational cost, which is common for advanced inference-time methods. The paper's honesty about this and other limitations is commendable. While the presentation could be polished for better clarity, the core ideas are well-justified and supported by strong empirical evidence. The work represents a meaningful step forward in building more capable, stateful reasoning systems with large language models.

---

# SID: Multi-LLM Debate Driven by Self Signals

Authors: Xuhang Chen, Zhifan Song, Deyi Ji, Shuo Gao, Lanyun Zhu

Keywords: Multi-agent debate, Large Language Models, Self signals, Model-level confidence, Token-level semantic focus, Early-exit mechanism, Adaptive compression, Efficiency optimization

Comments: None

Paper link: [http://arxiv.org/abs/2510.06843v1](http://arxiv.org/abs/2510.06843v1)

## Abstract

Large Language Models (LLMs) have exhibited impressive capabilities across diverse application domains. Recent work has explored Multi-LLM Agent Debate (MAD) as a way to enhance performance by enabling multiple LLMs to discuss and refine responses iteratively. Nevertheless, existing MAD methods predominantly focus on utilizing external structures, such as debate graphs, using LLM-as-a-Judge, while neglecting the application of self signals, such as token logits and attention, that arise during generation. This omission leads to redundant computation and potential performance degradation. In this paper, we shift the focus to the self signals of multi-LLM debate and introduce a Self-Signals Driven Multi-LLM Debate (SID), which leverages two types of self-signals: model-level confidence and token-level semantic focus, to adaptively guide the debate process. Our approach enables high-confidence agents to exit early at the model level and compress the redundant debate contents based on the attention mechanism. We evaluate our method on various LLMs and Multimodal LLMs across multiple challenging benchmarks. Experimental results demonstrate that our method not only outperforms existing MAD techniques in accuracy but also reduces token consumption, highlighting the effectiveness of utilizing self signals in enhancing both the performance and efficiency of multi-agent debate systems. Our code will be available at~\href{https://github.com/xuhang2019/SID}{\texttt{https://github.com/xuhang2019/SID}}.

## Summary

Here is a summary of the paper "SID: Multi-LLM Debate Driven by Self Signals":

**Key Contributions:** This paper introduces SID, a novel multi-agent debate framework that leverages internal "self signals" from LLMs during generation, rather than relying on external mechanisms like debate graphs or LLM-as-a-judge. The key innovation is using two types of internal signals: model-level confidence (from output probabilities) and token-level semantic focus (from attention patterns) to dynamically guide the debate process.

**Methods:** SID incorporates two main mechanisms: (1) An early-exit strategy using model-level confidence scores derived from token-wise uncertainty metrics (entropy and negative log-likelihood), allowing confident agents to skip unnecessary debate rounds via a vocabulary-adaptive threshold. (2) An adaptive compression mechanism that uses attention patterns conditioned on disagreement-oriented prompts to identify and retain semantically crucial debate content while compressing redundant information, significantly reducing token overhead.

**Results:** Extensive experiments across multiple LLMs (LLaMA3.1-8B, GPT-OSS-20B) and MLLMs (LLaVA1.6-13B, GLM4.1V) on benchmarks including MMLUpro, Math, ScienceQA, and MMStar demonstrate that SID consistently outperforms existing multi-agent debate methods in accuracy while achieving up to 40% reduction in token consumption. The framework shows strong scalability with additional debate rounds and maintains robust performance across different model types and task domains.

The work highlights the significant potential of leveraging internal model states for efficient and effective multi-agent collaboration, providing a new direction beyond structural optimizations in multi-LLM systems.

## Critique

Of course. Here is a commentary on the strengths and weaknesses of the paper "SID: Multi-LLM Debate Driven by Self Signals."

### Overall Assessment

This is a well-executed and impactful paper that addresses a clear and important problem in multi-agent debate (MAD) systems: the trade-off between performance gains and computational cost. The proposed method, SID, is novel, rigorously evaluated, and demonstrates significant improvements in both accuracy and efficiency.

---

### Strengths

**1. High Novelty and Conceptual Shift:**
The core contribution is a pivot from relying on "external" mechanisms (like LLM-as-a-judge or complex debate graphs) to leveraging "self signals" intrinsic to the LLM's generation process. Using token logits for confidence and attention maps for semantic focus is a clever and underexplored approach. This internal perspective is a fresh and valuable contribution to the field.

**2. Well-Designed, Two-Pronged Method:**
The paper elegantly tackles the efficiency problem from two complementary angles:
*   **Early-Exit (Model-Level):** This is a practical and intuitive idea. The "vocabulary-adaptive threshold" is a simple yet effective solution to the problem of comparing confidence across models with different vocabulary sizes.
*   **Adaptive Compression (Token-Level):** Using attention maps conditioned on a disagreement prompt to identify and preserve semantically critical parts of the debate is innovative. The `SemanticPreserve` heuristic to maintain coherent text units is a crucial detail that moves beyond naive token selection.

**3. Comprehensive and Convincing Evaluation:**
*   **Broad Scope:** The evaluation across multiple model types (LLMs and MLLMs) and diverse, challenging benchmarks (MMLUpro, Math, GPQA, etc.) strongly supports the generalizability of the method.
*   **Clear Metrics:** Reporting both accuracy and token consumption ratio directly addresses the paper's central thesis.
*   **Significant Results:** The results are impressive. Achieving state-of-the-art performance *while also* reducing token consumption by up to 40% is a compelling dual achievement that few papers in this area can claim.

**4. Excellent Analysis and Ablation:**
The paper goes beyond just presenting results. The ablation studies (Table 3) clearly demonstrate the contribution of each component. The analysis of hyperparameters like `α` and `p` (Figure 2e,f) provides practical guidance for future implementations and reinforces the robustness of the design choices.

**5. High Clarity and Presentation:**
The paper is exceptionally well-written and structured.
*   The problem is clearly motivated in the introduction.
*   The method (Section 4) is explained with sufficient technical detail, supported by a clear algorithm (Algorithm 1) and an illustrative framework diagram (Figure 1).
*   The use of case studies and visualizations (Figure 3) helps build an intuitive understanding of how SID works in practice.

---

### Weaknesses

**1. Limited Discussion of Computational Overhead:**
While the method reduces *token* consumption (a proxy for API cost), it introduces new computational steps: extracting logits for the entire sequence and running a forward pass to obtain attention maps for compression. A brief discussion of the latency/FLOPs overhead of these operations compared to the savings from early-exit and shorter contexts would provide a more complete picture of its efficiency.

**2. Dependence on Model Internals:**
The method's effectiveness is contingent on access to the model's internal states (logits and attention maps). This can be a limitation for closed-source models accessed via API (e.g., GPT-4), where such internals are typically unavailable. The paper implicitly assumes a more open or local deployment scenario.

**3. Potential Simplification in Confidence Estimation:**
The concatenation of eight different confidence metrics, while empirically effective, is somewhat heuristic. The finding that the simple vocabulary-adaptive threshold (SID-v) performs as well as the trained classifier (SID-c) is convenient for practicality but also suggests that a more theoretically grounded, single confidence metric might be possible.

**4. Scope of "Semantic Preservation":**
The `SemanticPreserve` heuristic, while necessary, is based on syntactic boundaries (commas, periods). Its ability to truly preserve complex, cross-sentence semantic meaning in all cases is not thoroughly evaluated. There is a risk that in highly nuanced debates, critical logical connections between compressed spans could be lost.

**5. Reproducibility (Minor):**
Although a reproducibility statement is included and the method is well-described, the actual release of the code (pointed to as "will be available") is crucial for the community to validate and build upon these results. The strength of this point is currently pending the code's release and quality.

---

### Summary

**SID** is a significant and timely contribution to the field of multi-agent systems. Its core novelty—using self-signals for efficient debate—is powerful and well-executed. The paper demonstrates substantial improvements in both performance and efficiency through rigorous experimentation. The weaknesses are relatively minor and primarily relate to the practical implementation details and scope of the evaluation, rather than the core concept, which is sound and impactful. This work successfully opens a promising new direction for research in efficient collaborative AI systems.

---

# The Markovian Thinker

Authors: Milad Aghajohari, Kamran Chitsaz, Amirhossein Kazemnejad, Sarath Chandar, Alessandro Sordoni, Aaron Courville, Siva Reddy

Keywords: Markovian Thinking, Reinforcement Learning, Chain-of-Thought, LongCoT, Delethink, Efficient Reasoning, Linear Compute, Context Scaling, Reasoning LLMs

Comments: None

Paper link: [http://arxiv.org/abs/2510.06557v1](http://arxiv.org/abs/2510.06557v1)

## Abstract

Reinforcement learning (RL) has recently become a strong recipe for training reasoning LLMs that produce long chains of thought (LongCoT). Yet the standard RL "thinking environment", where the state is the prompt plus all prior reasoning tokens, makes the state unbounded and forces attention-based policies to pay quadratic compute as thoughts lengthen. We revisit the environment itself. We propose Markovian Thinking, a paradigm in which the policy advances reasoning while conditioning on a constant-size state, decoupling thinking length from context size. As an immediate consequence this yields linear compute with constant memory. We instantiate this idea with Delethink, an RL environment that structures reasoning into fixed-size chunks. Within each chunk, the model thinks as usual; at the boundary, the environment resets the context and reinitializes the prompt with a short carryover. Through RL, the policy learns to write a textual state near the end of each chunk sufficient for seamless continuation of reasoning after reset. Trained in this environment, an R1-Distill 1.5B model reasons in 8K-token chunks yet thinks up to 24K tokens, matching or surpassing LongCoT-RL trained with a 24K budget. With test-time scaling, Delethink continues to improve where LongCoT plateaus. The effect of linear compute is substantial: we empirically estimate at 96K average thinking length LongCoT-RL costs 27 H100-months vs. 7 for Delethink. Analysis at RL initialization shows off-the-shelf reasoning models (1.5B-120B) often sample Markovian traces zero-shot across diverse benchmarks, providing positive samples that make RL effective at scale. Our results show that redesigning the thinking environment is a powerful lever: it enables very long reasoning without quadratic overhead and opens a path toward efficient, scalable reasoning LLMs.

## Summary

Based on the provided paper, here is a summary focusing on its key contributions, methods, and results:

**Key Contributions:**
This paper introduces the "Markovian Thinking" paradigm and its practical instantiation, "Delethink," a novel Reinforcement Learning (RL) framework designed to train reasoning Large Language Models (LLMs). The core innovation is to decouple the total thinking length (the number of reasoning tokens) from the model's context window size. By structuring the RL environment to operate in fixed-size chunks, Delethink achieves linear computational scaling and constant memory usage with respect to thinking length, a significant improvement over the quadratic scaling of standard LongCoT (Long Chain-of-Thought) RL methods.

**Methods:**
The Delethink method reformulates the standard RL environment for reasoning LLMs. Instead of allowing the context (the prompt plus all prior reasoning tokens) to grow unbounded, Delethink forces the model to reason in a sequence of fixed-size chunks (e.g., 8K tokens). At the end of each chunk, the environment resets the context, and the next chunk's prompt is constructed from the original query plus only a short "Markovian state" (e.g., the last 4K tokens) from the previous chunk. Through RL training, the policy learns to write a sufficient textual state at the end of each chunk to seamlessly continue reasoning after the reset. This design ensures the model's effective context size remains constant, avoiding the quadratic cost of self-attention over ever-longer sequences.

**Results:**
The authors demonstrate that an R1-Distill 1.5B model trained with Delethink (using an 8K chunk size but a 24K total thinking budget) matches or surpasses the performance of a model trained with standard LongCoT-RL under the same 24K budget on math benchmarks like AIME and HMMT. Crucially, Delethink shows superior **test-time scaling**, continuing to improve in accuracy when allowed to reason far beyond its training budget (e.g., up to 128K tokens), whereas LongCoT models plateau. The computational benefits are substantial: empirical estimates show that training for an average of 94K thinking tokens would cost 27 H100-months with LongCoT versus only 7 with Delethink. The paper also provides evidence that state-of-the-art reasoning LLMs (up to 120B parameters) already exhibit latent "Markovian Thinking" capabilities zero-shot, indicating that the approach is compatible with and can scale to powerful modern models.

## Critique

Of course. Here is a critique of the paper "The Markovian Thinker," focusing on its strengths, weaknesses, and overall contribution.

### Overall Summary

This is a highly impressive and significant paper that tackles a fundamental bottleneck in scaling reasoning for large language models (LLMs). The core idea—decoupling thinking length from computational context—is both novel and impactful. The empirical results are strong, demonstrating that the proposed "Delethink" method can match or surpass the performance of standard "LongCoT" training while offering dramatic computational savings and superior test-time scaling. The presentation is generally clear, though dense in parts due to the technical nature of the contribution.

---

### Strengths

1.  **High Novelty and Paradigm Shift:** The paper's greatest strength is its conceptual novelty. Instead of trying to optimize *within* the existing "LongCoT" paradigm (which has a quadratic compute cost), it fundamentally redefines the underlying Reinforcement Learning (RL) environment. The shift from an ever-growing context to a "Markovian Thinking" paradigm with a fixed-size state is a powerful and elegant insight.

2.  **Significant and Well-Supported Results:** The empirical evaluation is comprehensive and compelling.
    *   **Performance Parity/Superiority:** The authors convincingly show that Delethink (trained with an 8K context) can perform as well as or better than LongCoT-RL (trained with a 24K context) on a range of mathematical and coding benchmarks. This directly validates the core claim.
    *   **Computational Efficiency:** The theoretical analysis of linear vs. quadratic scaling is backed by empirical measurements of FLOPs, memory, and throughput, making a strong case for the practical utility of the method.
    *   **Superior Test-Time Scaling:** A key finding is that Delethink models continue to improve when allowed to think for far longer than their training budget, whereas LongCoT models plateau. This suggests Delethink learns a more general and scalable reasoning strategy.

3.  **Compelling Analysis of "Why It Works":** The investigation into why Delethink is effective is a major strength. The finding that state-of-the-art reasoning LLMs (from 1.5B to 120B parameters) already exhibit strong "Markovian Thinking" capabilities *zero-shot* is surprising and crucial. It explains why RL training can succeed—the desired behavior is already in-distribution, providing a strong initialization.

4.  **Clear Practical Implications:** The paper effectively argues that this work paves the way for "million-token" reasoning by breaking the quadratic cost barrier. It also rightly points out the exciting implication for non-quadratic architectures (e.g., Mamba, linear attention), suggesting they might be particularly well-suited for reasoning tasks.

---

### Weaknesses and Areas for Improvement

1.  **Clarity of the RL Formulation and Derivation:** While the high-level idea is clear, the technical details of the modified Markov Decision Process (MDP) and the policy gradient derivation are somewhat glossed over. The transition function is stated, but its integration into the standard RL-for-LLMs framework (like PPO or GRPO) could be explained more intuitively. Relying on the appendix for the loss derivation makes the main text feel slightly incomplete.

2.  **Limited Exploration of the "Markovian State":** The paper makes a simple and effective choice for the Markovian state (the last `m` tokens), but it remains a black box. A deeper analysis of *what* the model learns to write into this carryover state would be fascinating. Does it learn to create summaries, store key variables, or use a special syntax? Understanding this could lead to further improvements.

3.  **Ablation and Hyperparameter Sensitivity:** While there is an ablation on context size (`C`), the paper would be strengthened by a more systematic study of the Markovian state size (`m`) and the iteration cap (`I`). The choice of `m = C/2` seems somewhat arbitrary, and understanding the sensitivity of performance to this ratio would be valuable for practitioners.

4.  **Stress Testing on Truly Long-Range Dependencies:** The stress test on CrossWordBench is a good start, but it only scratches the surface. The method's main limitation is its inherent assumption that the "summary" of all past reasoning can fit into `m` tokens. Tasks that require holding many independent facts or long-range dependencies in active memory (e.g., complex multi-step planning or narrative reasoning) might be a fundamental challenge for this approach. Demonstrating performance on such tasks, or clearly outlining this as a limitation, would provide a more complete picture.

---

### Conclusion

"The Markovian Thinker" is a top-tier contribution that introduces a paradigm-shifting approach to efficient reasoning in LLMs. Its strengths—a novel concept, significant empirical results, and compelling analysis—far outweigh its weaknesses. The paper is likely to have a high impact, inspiring new research directions in efficient RL training, long-context reasoning, and the application of linear-time architectures. The work is not just an incremental improvement but a foundational step towards making extremely long-chain reasoning practically feasible.

---

# Evolving and Executing Research Plans via Double-Loop Multi-Agent Collaboration

Authors: Zhi Zhang, Yan Liu, Zhejing Hu, Gong Chen, Sheng-hua Zhong, Jiannong Cao

Keywords: multi-agent collaboration, automated scientific research, double-loop learning, evolutionary algorithms, research plan generation

Comments: None

Paper link: [http://arxiv.org/abs/2510.06761v1](http://arxiv.org/abs/2510.06761v1)

## Abstract

Automating the end-to-end scientific research process poses a fundamental challenge: it requires both evolving high-level plans that are novel and sound, and executing these plans correctly amidst dynamic and uncertain conditions. To address this bilevel challenge, we propose a novel Double-Loop Multi-Agent (DLMA) framework to solve the given research problem automatically. The leader loop, composed of professor agents, is responsible for evolving research plans. It employs an evolutionary algorithm through involvement, improvement, and integration meetings to iteratively generate and refine a pool of research proposals, exploring the solution space effectively. The follower loop, composed of doctoral student agents, is responsible for executing the best-evolved plan. It dynamically adjusts the plan during implementation via pre-hoc and post-hoc meetings, ensuring each step (e.g., drafting, coding) is well-supported by contextual and external observations. Extensive experiments on benchmarks like ACLAward and Laboratory show that DLMA generates research papers that achieve state-of-the-art scores in automated evaluation, significantly outperforming strong baselines. Ablation studies confirm the critical roles of both loops, with evolution driving novelty and execution ensuring soundness.

## Summary

This paper introduces the **Double-Loop Multi-Agent (DLMA) Framework** for automating the end-to-end scientific research process. The core challenge addressed is the bilevel optimization problem in research: "doing the right things" (evolving novel and sound high-level plans) and "doing things right" (executing these plans correctly amid dynamic conditions).

**Key Contributions:**
- **DLMA Framework**: A novel two-loop architecture:
  - **Leader Loop**: Composed of "professor" agents that evolve research plans through an evolutionary algorithm with three meeting types—*involvement* (introducing diverse perspectives from references), *improvement* (refining proposals by addressing weaknesses), and *integration* (combining strengths of different proposals). This loop explores the solution space to generate optimal plans.
  - **Follower Loop**: Composed of "doctoral student" agents that execute the best-evolved plan. It dynamically adjusts actions via *pre-hoc* (before action, using contextual/external observations) and *post-hoc* (after action, updating subsequent steps) meetings to ensure alignment and correctness.

**Methods:**
- The leader loop iteratively refines a population of proposals, selecting top candidates via a simulated review panel.
- The follower loop executes plans step-by-step, drafting sections, managing code projects, and ensuring consistency through iterative revisions based on observations and execution logs.
- Evaluated on benchmarks like **ACLAward** and **Laboratory** using LLM-as-judge with ACL-style review criteria (Soundness, Excitement, Overall, Confidence).

**Key Results:**
- DLMA achieves **state-of-the-art scores** on both datasets, outperforming strong baselines (e.g., GPT-5, Gemini 2.5 Pro, Claude Sonnet 4) and multi-agent frameworks (CycleResearcher, Agent Laboratory, Dolphin).
- **Ablation studies** confirm both loops are critical: evolution drives novelty/excitement, while execution ensures soundness/technical solidity.
- **Case studies** show DLMA-generated papers align with human expert work in identifying techniques and proposing solutions, though complex coding remains a limitation.
- The framework demonstrates effective plan-evolution (improving mean scores over generations) and dynamic execution (maintaining high support rates between observations and plans).

**Limitations:** High computational cost (~1,558 seconds, ~1.75M tokens) and occasional code hallucination. Future work will focus on paper-code alignment for more reliable experiments.

## Critique

Of course. Here is a critique of the paper "Evolving and Executing Research Plans via Double-Loop Multi-Agent Collaboration."

### Strengths

1.  **Novel Conceptual Framework:** The paper's core strength is its well-motivated and novel conceptual framework. Framing automated scientific research as a **bilevel optimization problem** is elegant and provides a clear mathematical foundation. The analogy to **double-loop learning** from organizational theory is insightful and effectively justifies the two-tiered (leader/follower) agent structure. This is a more sophisticated and principled approach than many existing multi-agent systems that simply chain together specialized agents.

2.  **Comprehensive and Ambitious Evaluation:** The authors conduct a thorough evaluation across multiple benchmarks (ACLAward, Laboratory, Plagiarism). Using award-winning papers as a high bar for comparison is ambitious and lends credibility to their claims. The use of multiple, standardized review forms (ACL, ICLR, NeurIPS) for LLM-as-a-judge evaluation helps to reduce bias and demonstrates the robustness of their results.

3.  **Insightful Ablation and Analysis:** The paper goes beyond simply reporting superior results. The ablation study clearly delineates the contributions of the "evolution" (leader) and "adaptation" (follower) loops, showing that the former drives novelty/excitement while the latter ensures soundness. The additional analyses in Sections 4.4 and 4.5 provide valuable insights into the internal dynamics of the system, such as the performance plateau of the evolution process and the increasing need for plan adaptation in later research stages.

4.  **Clear Presentation of Methodology:** The methodology section is generally well-structured. The problem formulation is precise, and the descriptions of the leader loop (with its three meeting types) and the follower loop (with its pre/post-hoc meetings) are detailed enough to convey the core mechanics of the system. The use of mathematical notation aids in clarity.

### Weaknesses

1.  **Significant Computational Cost:** The most glaring weakness is the enormous computational cost, which the authors transparently acknowledge in the limitations. A cost of **~1,558 seconds and ~1.75 million tokens** per research paper is prohibitively expensive for most practical applications. This raises serious questions about the scalability and economic viability of the approach, positioning it more as a proof-of-concept than a readily deployable tool.

2.  **Over-reliance on LLM-as-a-Judge:** While LLM-based evaluation is a pragmatic and reproducible choice, it is an imperfect proxy for human judgment. The reported Spearman correlation of 0.46 with human experts, while positive, indicates only a moderate alignment. The paper would be significantly stronger with a more extensive human evaluation, especially to validate the claimed "state-of-the-art" performance in terms of genuine scientific novelty and correctness, which are difficult for current LLMs to assess reliably.

3.  **Lack of Technical Depth on Implementation:** The paper describes the *what* but often glosses over the *how*. Key implementation details are missing:
    *   **Prompting Strategies:** The prompts used for the various "meetings" (involvement, improvement, integration, pre/post-hoc) are critical to the system's performance but are not provided or discussed in detail.
    *   **Observation Retrieval:** The method for fetching "contextual and external observations" is described at a high level but lacks specifics on the retrieval models, ranking, or how relevance is determined, which is a non-trivial challenge.
    *   **Code Agent Hallucination:** The limitation regarding the code agent "hallucinating" is a major issue for a system aiming to produce sound research. The paper does not detail the extent of this problem or the specific measures taken to mitigate it beyond using IterativeAgent.

4.  **Ambiguity in "Solving" Research Problems:** The paper claims the system "solves" research problems, but the output is a generated paper, not necessarily a validated scientific finding. The true test of "solving" a problem would involve implementing the proposed method and empirically verifying its claims against a ground truth, which is not done here. The evaluation is on the *narrative and structure* of the paper, not the veracity of its scientific content.

### Overall Assessment

This is a highly ambitious and conceptually novel paper that makes a valuable contribution to the field of AI-driven scientific discovery. The double-loop framework is a significant architectural advance over existing multi-agent systems. The results demonstrate a clear performance improvement on automated metrics.

However, the work is hampered by its extreme computational cost and a reliance on automated evaluation that cannot fully capture scientific rigor. It serves as a compelling blueprint for future research rather than a finished solution. Addressing the limitations of cost, implementing more robust validation (both human and empirical), and providing deeper technical details would be crucial next steps.

---

# Think Natively: Unlocking Multilingual Reasoning with Consistency-Enhanced Reinforcement Learning

Authors: Xue Zhang, Yunlong Liang, Fandong Meng, Songming Zhang, Kaiyu Huang, Yufeng Chen, Jinan Xu, Jie Zhou

Keywords: Multilingual Reasoning, Reinforcement Learning, Language Consistency, Cross-lingual Alignment, Large Reasoning Models

Comments: 13 pages, 8 tables, 4 figures

Paper link: [http://arxiv.org/abs/2510.07300v1](http://arxiv.org/abs/2510.07300v1)

## Abstract

Large Reasoning Models (LRMs) have achieved remarkable performance on complex reasoning tasks by adopting the "think-then-answer" paradigm, which enhances both accuracy and interpretability. However, current LRMs exhibit two critical limitations when processing non-English languages: (1) They often struggle to maintain input-output language consistency; (2) They generally perform poorly with wrong reasoning paths and lower answer accuracy compared to English. These limitations significantly degrade the user experience for non-English speakers and hinder the global deployment of LRMs. To address these limitations, we propose M-Thinker, which is trained by the GRPO algorithm that involves a Language Consistency (LC) reward and a novel Cross-lingual Thinking Alignment (CTA) reward. Specifically, the LC reward defines a strict constraint on the language consistency between the input, thought, and answer. Besides, the CTA reward compares the model's non-English reasoning paths with its English reasoning path to transfer its own reasoning capability from English to non-English languages. Through an iterative RL procedure, our M-Thinker-1.5B/7B models not only achieve nearly 100% language consistency and superior performance on two multilingual benchmarks (MMATH and PolyMath), but also exhibit excellent generalization on out-of-domain languages.

## Summary

Here is a summary of the paper "Think Natively: Unlocking Multilingual Reasoning with Consistency-Enhanced Reinforcement Learning":

**Key Contributions:**
The paper introduces M-Thinker, a framework designed to address two critical limitations in Large Reasoning Models (LRMs) when processing non-English languages: (1) inability to maintain input-output language consistency, and (2) inferior reasoning performance compared to English. The authors propose a novel reinforcement learning approach that combines strict language consistency constraints with cross-lingual reasoning alignment to create truly multilingual reasoning models.

**Methods:**
The core methodology employs Group Relative Policy Optimization (GRPO) with two key reward components: a Language Consistency (LC) reward that strictly enforces input-output language alignment, and a Cross-lingual Thinking Alignment (CTA) reward that uses the model's own English reasoning paths as references to improve reasoning quality in other languages. The training procedure involves cold-start supervised fine-tuning, rejection sampling to select challenging but solvable problems, and iterative RL training. The approach was evaluated on two multilingual mathematical reasoning benchmarks (MMATH and PolyMath) across ten languages.

**Results:**
M-Thinker achieves remarkable improvements over baseline methods. The models achieve nearly 100% language consistency on in-domain languages while significantly improving answer accuracy. For example, M-Thinker-7B outperforms the original DeepSeek-R1-0528 model on the combined LC&Acc metric across multiple languages. The method also demonstrates excellent generalization to out-of-domain languages, with performance patterns suggesting better transfer within similar language families. The ablation studies confirm the importance of both the LC and CTA rewards, with the CTA reward being particularly dependent on the quality of the judge model used.

## Critique

Based on the paper "Think Natively: Unlocking Multilingual Reasoning with Consistency-Enhanced Reinforcement Learning," here is an analysis of its strengths, weaknesses, and overall contribution:

### Strengths

1. **Novelty of Approach**:
   - The introduction of **Cross-lingual Thinking Alignment (CTA) reward** is a key innovation. It leverages the model's own English reasoning paths as a "teacher" to improve multilingual reasoning, which is a clever way to address performance disparities between English and non-English languages.
   - The **Language Consistency (LC) reward** enforces strict input-output language consistency, addressing a critical limitation of existing Large Reasoning Models (LRMs) in multilingual settings. The combination of LC and CTA rewards is a well-motivated solution.
   - The **iterative RL training procedure** incorporating cold-start SFT, rejection sampling, and GRPO is systematic and demonstrates a thoughtful design to progressively enhance the model's multilingual reasoning capabilities.

2. **Significance of Results**:
   - The results are highly impressive: M-Thinker achieves **nearly 100% language consistency** on in-domain languages while significantly improving accuracy on multilingual benchmarks like MMATH and PolyMath.
   - The model demonstrates **strong generalization to out-of-domain languages**, which is crucial for real-world deployment. The fact that M-Thinker-7B even outperforms the larger DeepSeek-R1-0528 on LC&Acc for several languages is a notable achievement.
   - The ablation studies convincingly validate the contributions of each component (LC reward, CTA reward, cold-start SFT, etc.), strengthening the paper's claims.

3. **Clarity of Presentation**:
   - The paper is well-structured, with clear sections for methodology, experiments, and analysis. The use of tables to summarize results makes the comparisons easy to follow.
   - The problem statement is clearly articulated, and the limitations of existing methods (e.g., trade-off between language consistency and accuracy) are well-motivated.
   - The inclusion of a detailed training procedure (Algorithm 1) and reward formulations enhances reproducibility.

### Weaknesses

1. **Scope and Scalability**:
   - The experiments are limited to **five in-domain languages** and two model sizes (1.5B and 7B). While the results are promising, it is unclear how well the approach scales to a larger set of languages or significantly larger models.
   - The reliance on **synthetic data** (translated via DeepSeek-V3) for training may introduce biases or inaccuracies, which are not thoroughly discussed.

2. **Dependency on External Tools**:
   - The **language consistency reward** depends on the `langdetect` library, which may not be fully robust for all languages or mixed-language texts. The paper acknowledges this but does not explore alternatives.
   - The **CTA reward** relies on an external judge model (DeepSeek-V3), which introduces computational overhead and potential biases. The performance drop when using a weaker judge model (Qwen2.5-7B) highlights this dependency.

3. **Generalization Analysis**:
   - While the paper includes a generalization study, it is relatively limited. The observation that models generalize better to languages within the same family is intuitive but not deeply explored. A more rigorous linguistic analysis could strengthen this section.

4. **Reproducibility and Cost**:
   - The iterative RL training procedure, rejection sampling, and reliance on a strong judge model (DeepSeek-V3) make the approach computationally expensive and less accessible for researchers with limited resources.

### Overall Assessment

This paper presents a highly innovative and effective solution to a critical problem in multilingual reasoning. The combination of LC and CTA rewards, along with a well-designed training pipeline, addresses both language consistency and reasoning accuracy simultaneously. The results are state-of-the-art on standard benchmarks, and the approach demonstrates strong generalization.

However, the scalability to more languages and larger models, dependency on external tools, and computational cost are areas that warrant further investigation. Despite these limitations, the paper makes a significant contribution to multilingual reasoning and provides a strong foundation for future work.

