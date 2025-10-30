---
title: "ArXiv Daily Digest on 2025-10-29"
collection: digests
permalink: /digests/arxiv_cs_CL_2025-10-29_report
date: 2025-10-29
location: "Online"
---

Today's research landscape showcases significant advancements in multi-agent AI systems, with several papers exploring how Large Language Model (LLM) agents can collaborate through sophisticated reasoning and communication frameworks. The papers collectively demonstrate a shift toward outcome-supervised training paradigms that incentivize autonomous exploration, particularly in knowledge-intensive tasks like Knowledge Base Question Answering (KBQA). Notable innovations include graph-based planning for parallel tool execution in GAP (Graph-based Agent Planning), socio-cognitive evaluation frameworks for proactive mediators in ProMediate, and large-scale empirical benchmarks like DEBATE for assessing authentic multi-agent dynamics. These works highlight growing emphasis on efficient collaboration under information asymmetry, with reinforcement learning (RL) and curriculum learning emerging as key techniques for developing more robust and human-aligned AI systems.

## TL;DR

Total papers: 54 , Selected papers: 5

**TL;DR: Recent Research on Multi-Agent AI Systems**

This collection of papers demonstrates significant advances in **multi-agent collaboration and planning**, with a focus on improving efficiency, reasoning capabilities, and social intelligence in complex interactive settings.

**Key Themes:**

1. **Graph-Based Planning for Efficiency**  
   - **GAP** (2510.25320) introduces graph-based dependency modeling to enable parallel tool execution, reducing LLM interaction turns by 33.4% and execution time by 20% while improving multi-hop reasoning accuracy.  
   - **KnowCoder-A1** (2510.25101) uses outcome-only supervision and curriculum RL to train KBQA agents that achieve 11.1% improvement on zero-shot tasks with 12x less data.

2. **Communication Strategies Under Information Asymmetry**  
   - **Communication and Verification** (2510.25595) shows agents with both information-seeking and -providing capabilities achieve highest success rates in collaborative puzzles, while environment-based verifiers improve performance by 30+ percentage points.

3. **Socio-Cognitive Evaluation Frameworks**  
   - **ProMediate** (2510.25224) provides the first systematic testbed for proactive AI mediators in multi-party negotiations, with socially intelligent mediators achieving 3.6 percentage points higher consensus change while responding 77% faster in hard scenarios.
   - **DEBATE** (2510.25110) introduces a large-scale benchmark (29,417 messages) revealing that LLM-simulated groups exhibit premature consensus and systematic behavioral differences from human dynamics.

**Insight:** Current multi-agent systems excel at optimizing efficiency and task performance but struggle with authentic human-like social dynamics, highlighting the need for better socio-cognitive alignment beyond surface-level metrics.

**Papers:**
- https://arxiv.org/abs/2510.25101
- https://arxiv.org/abs/2510.25320  
- https://arxiv.org/abs/2510.25595
- https://arxiv.org/abs/2510.25224
- https://arxiv.org/abs/2510.25110

---

# KnowCoder-A1: Incentivizing Agentic Reasoning Capability with Outcome Supervision for KBQA

Authors: Zhuo Chen, Fei Wang, Zixuan Li, Zhao Zhang, Weiwei Ding, Chuanguang Yang, Yongjun Xu, Xiaolong Jin, Jiafeng Guo

Keywords: Knowledge Base Question Answering, Agentic Reasoning, Outcome Supervision, Reinforcement Learning, Curriculum Learning, Multi-stage Training, Autonomous Exploration

Comments: None

Paper link: [http://arxiv.org/abs/2510.25101v1](http://arxiv.org/abs/2510.25101v1)

## Abstract

Knowledge Base Question Answering (KBQA) aims to answer natural-language questions over a structured Knowledge Base (KB). Recent work improves KBQA by adopting an agentic reasoning paradigm, in which Large Language Models (LLMs) iteratively decompose a question, generate its corresponding logical queries, and interact with the KB to derive the answer. However, these methods typically fine-tune LLMs on reasoning trajectories synthesized via process supervision, which offers weak incentives for exploration and thus fails to strengthen the agentic reasoning ability. In this paper, we propose KnowCoder-A1, an LLM that can autonomously perform agentic reasoning on KBs to obtain answers. To incentivize autonomous exploration, KnowCoder-A1 trains the LLM under outcome-only supervision via a multi-stage curriculum reinforcement learning with an easy-to-hard curriculum. To establish foundational agentic capabilities, KnowCoder-A1 first fine-tunes the LLM on a small set of high-quality trajectories obtained through outcome-based rejection sampling. Then, to alleviate the reward sparsity inherent in outcome-only supervision, it applies multi-stage curriculum RL with reward schedules that progress from easy to hard. Trained with outcome-only supervision, KnowCoder-A1 exhibits powerful reasoning behaviors and consistently outperforms prior approaches across three mainstream datasets. Notably, on the zero-shot subset of GrailQA, KnowCoder-A1 achieves up to an 11.1% relative improvement while using only one-twelfth of the training data, demonstrating strong agentic reasoning capabilities.

## Summary

Based on the provided paper, here is a summary of "KnowCoder-A1: Incentivizing Agentic Reasoning Capability with Outcome Supervision for KBQA":

**Key Contributions:** The paper introduces KnowCoder-A1, a novel framework for Knowledge Base Question Answering (KBQA) that moves away from traditional process-supervised methods. Its main contribution is a training paradigm that relies primarily on *outcome-only supervision* to incentivize autonomous exploration in agentic reasoning, addressing the limitations of existing methods that rely on pre-defined, idealized reasoning trajectories. This approach aims to enhance the model's robustness (ability to recover from errors) and flexibility (ability to discover diverse correct reasoning paths).

**Methods:** The proposed method is a multi-stage training framework. First, a **cold-start stage** uses Supervised Fine-Tuning (SFT) on a small, high-quality dataset. This dataset is curated not from decomposing gold logical forms, but via *outcome-based rejection sampling*, where strong LLMs generate diverse trajectories that are filtered based solely on whether they lead to the correct final answer. Second, a **Reinforcement Learning (RL) exploration stage** employs Group Relative Policy Optimization (GRPO) with a carefully designed *easy-to-hard curriculum*. This curriculum uses a composite reward: it starts with a precision-focused F0.5 score to encourage finding correct answers (the "easy" phase) and then transitions to a balanced F1 score to incentivize finding all correct answers (the "hard" phase), effectively mitigating reward sparsity.

**Results:** Extensive experiments on three mainstream KBQA datasets (WebQSP, CWQ, and GrailQA) demonstrate that KnowCoder-A1 consistently outperforms previous state-of-the-art agentic KBQA methods. Notably, on the challenging zero-shot subset of GrailQA, it achieves a relative improvement of up to 11.1% over the previous best method (KBQA-o1). Crucially, it achieves these superior results while using significantly less training data (12x less than KBQA-o1 on GrailQA) and a more efficient single-pass inference, avoiding the computational cost of methods like MCTS. Ablation studies confirm the necessity of both the RL stage and the easy-to-hard reward curriculum.

## Critique

Of course. Here is a critique of the paper "KnowCoder-A1: Incentivizing Agentic Reasoning Capability with Outcome Supervision for KBQA," focusing on its strengths and weaknesses.

### Summary of Strengths

1.  **Well-Motivated and Novel Approach:** The paper identifies a clear and important limitation in existing agentic KBQA methods: their reliance on "process supervision" (training on pre-defined, idealized reasoning trajectories). The proposed shift to "outcome-only supervision" is a significant and timely contribution, aligning with trends in RL that seek to reduce heavy reliance on expert demonstrations. The core idea—that rewarding only the final correct answer incentivizes more robust and flexible exploration—is both intuitive and powerful.

2.  **Comprehensive and Compelling Results:** The empirical evaluation is a major strength. The authors demonstrate state-of-the-art performance on three major benchmarks (WebQSP, CWQ, GrailQA) while using significantly less training data (12x less than the previous SOTA on GrailQA). The standout improvement on the "zero-shot" subset of GrailQA (an 11.1% relative gain) is a strong validation of the method's generalization capabilities, which is a primary goal of moving away from rigid process supervision.

3.  **Effective Multi-Stage Training Design:** The proposed methodology is well-structured and logically sound. The "cold-start" SFT stage using high-quality, outcome-filtered trajectories is a pragmatic solution to bootstrap the agent, avoiding the instability of training from scratch with RL. The two-phase RL curriculum (F0.5 -> F1 reward) is a clever and well-justified mechanism to combat reward sparsity and hacking, guiding the agent from broad exploration to precise execution.

4.  **Strong and Insightful Analysis:** The paper goes beyond just reporting final scores. The ablation studies (Sec 5.3) convincingly demonstrate the necessity of each component (the curriculum, the RL stage, and the data upsampling). The analysis of training dynamics (Sec 5.4.1) and the evolution of robustness and flexibility (Sec 5.4.2) provides valuable insights into *how* and *why* the method works, showing a clear learning progression from exploration to efficient exploitation.

### Summary of Weaknesses

1.  **Clarity and Presentation of the Reward Curriculum:** While the concept of the reward curriculum is excellent, its explanation could be clearer. The transition criterion between Phase 1 (F0.5) and Phase 2 (F1) is not explicitly stated. Is it based on a performance plateau, a fixed number of steps, or a manual switch? A more precise description of this scheduling would improve reproducibility.

2.  **Limited Discussion of Computational Cost:** The paper rightly highlights the inference-time efficiency of their single-pass model compared to MCTS-based methods. However, it does not discuss the computational cost of the RL training stage itself. Generating multiple rollouts for each question during RL can be expensive. A brief discussion of the total training time or computational budget compared to the SFT baselines would provide a more complete picture of the method's efficiency.

3.  **Potential Overstatement of "Zero Process Supervision":** The claim of using "zero process-supervised instances" is technically true for the RL stage. However, the cold-start SFT stage relies entirely on trajectories that, while generated via outcome-sampling, still represent *processes*. The model is still initially biased towards the reasoning styles present in these demos. The paper could more candidly acknowledge that the "process" bias is shifted from human-designed trajectories to LLM-generated ones, rather than eliminated entirely.

4.  **Narrowed Scope of Baselines:** The comparison is primarily focused on other "low-resource" or agentic methods. While this is fair, including a top-performing, fully-supervised semantic parsing model (even if just as a reference point in the introduction or related work) would better contextualize the performance gap that remains between low-resource agentic methods and models trained on large-scale, annotated logical forms.

### Overall Assessment

This is a strong paper that makes a meaningful contribution to the field of KBQA and agentic reasoning. The core idea is novel and well-executed, the experimental results are impressive and thoroughly analyzed, and the methodology is sound. The weaknesses are primarily related to the clarity of certain implementation details and the scope of discussion, rather than fundamental flaws in the approach. The work successfully demonstrates that outcome-only supervision, when combined with a thoughtful curriculum, can foster more robust and generalizable reasoning agents than traditional process-supervised fine-tuning.

---

# GAP: Graph-Based Agent Planning with Parallel Tool Use and Reinforcement Learning

Authors: Jiaqi Wu, Qinlao Zhao, Zefeng Chen, Kai Qin, Yifei Zhao, Xueqian Wang, Yuhang Yao

Keywords: Graph-based Planning, Multi-Agent Systems, Tool-Integrated Reasoning, Parallel Tool Execution, Reinforcement Learning, Multi-Hop Question Answering

Comments: None

Paper link: [http://arxiv.org/abs/2510.25320v1](http://arxiv.org/abs/2510.25320v1)

## Abstract

Autonomous agents powered by large language models (LLMs) have shown impressive capabilities in tool manipulation for complex task-solving. However, existing paradigms such as ReAct rely on sequential reasoning and execution, failing to exploit the inherent parallelism among independent sub-tasks. This sequential bottleneck leads to inefficient tool utilization and suboptimal performance in multi-step reasoning scenarios. We introduce Graph-based Agent Planning (GAP), a novel framework that explicitly models inter-task dependencies through graph-based planning to enable adaptive parallel and serial tool execution. Our approach trains agent foundation models to decompose complex tasks into dependency-aware sub-task graphs, autonomously determining which tools can be executed in parallel and which must follow sequential dependencies. This dependency-aware orchestration achieves substantial improvements in both execution efficiency and task accuracy. To train GAP, we construct a high-quality dataset of graph-based planning traces derived from the Multi-Hop Question Answering (MHQA) benchmark. We employ a two-stage training strategy: supervised fine-tuning (SFT) on the curated dataset, followed by reinforcement learning (RL) with a correctness-based reward function on strategically sampled queries where tool-based reasoning provides maximum value. Experimental results on MHQA datasets demonstrate that GAP significantly outperforms traditional ReAct baselines, particularly on multi-step retrieval tasks, while achieving dramatic improvements in tool invocation efficiency through intelligent parallelization. The project page is available at: https://github.com/WJQ7777/Graph-Agent-Planning.

## Summary

Based on the provided paper, here is a summary focusing on its key contributions, methods, and results.

**Key Contributions:**
This paper introduces GAP (Graph-based Agent Planning), a novel training paradigm for LLM-based agents that addresses the inefficiency of sequential tool-use frameworks like ReAct. The core innovation is enabling agents to perform dependency-aware planning by explicitly modeling tasks as Directed Acyclic Graphs (DAGs). This allows the model to identify which sub-tasks can be executed in parallel and which must be run sequentially, combining the learnability of single-agent Tool-Integrated Reasoning (TIR) models with the expressive power of multi-agent coordination, but without the associated overhead.

**Methods:**
The GAP framework operates in three main phases:
1.  **Graph-based Task Decomposition:** The model analyzes a complex query to identify atomic sub-tasks and then constructs a dependency graph where nodes are sub-tasks and edges represent execution dependencies.
2.  **Dependency-Aware Execution:** The graph is partitioned into execution levels via topological sorting. All independent sub-tasks within the same level are executed in parallel, while dependent tasks are executed sequentially across levels.
3.  **Two-Stage Training Pipeline:** The model is first trained via Supervised Fine-Tuning (SFT) on a high-quality, synthetically generated dataset of 7,000 graph-based planning traces. This is followed by Reinforcement Learning (RL) using a correctness-based reward function to further optimize the policy for efficient and effective planning.

**Results:**
Extensive experiments on seven question-answering benchmarks demonstrate that GAP achieves a superior balance of accuracy and efficiency:
*   **Superior Performance:** GAP outperforms state-of-the-art baselines, showing an average performance improvement of 0.9% on complex multi-hop reasoning tasks.
*   **Enhanced Efficiency:** The framework significantly reduces the number of LLM interaction turns (by up to 33.4%), shortens response length (by ~25%), and lowers overall execution time (by over 20%) compared to sequential baselines like Search-R1. This leads to a better performance-cost trade-off, making multi-hop reasoning more practical for real-world deployment.

## Critique

Of course. Here is a critique of the paper "GAP: Graph-based Agent Planning with Parallel Tool Use and Reinforcement Learning."

### Summary of Strengths

1.  **Novel and Well-Motivated Core Idea:** The central premise—training an LLM to perform explicit graph-based dependency planning to enable parallel tool execution—is highly novel and addresses a clear, recognized bottleneck in sequential agent frameworks like ReAct. The insight that many sub-tasks in complex reasoning are independent is powerful and well-justified.

2.  **Comprehensive and Rigorous Methodology:** The paper presents a complete pipeline, from data synthesis and supervised fine-tuning (SFT) to reinforcement learning (RL). The use of a multi-agent system (Chain-of-Agents) with GPT-4o to generate high-quality training data is a sophisticated and credible approach. The explicit filtering criteria for the SFT data (complexity, diversity, length) demonstrate a careful and thoughtful curation process.

3.  **Strong and Multifaceted Evaluation:** The evaluation goes beyond just accuracy (Exact Match). The analysis of efficiency metrics—**interaction turns, response length, and inference time**—is a major strength, as it directly validates the paper's core claim of improved efficiency. The use of the "cost-of-pass" metric provides a practical, deployment-oriented perspective. Showing improvements across multiple in-domain and out-of-domain datasets demonstrates robust generalization.

4.  **Clear Presentation of the Paradigm:** The problem formulation (Section 3.1) is clear, and the breakdown of the GAP process into "Sub-task Identification," "Dependency Analysis," and "Graph Construction" is logical and easy to follow. The inclusion of a structured format for the graph output and Algorithm 1 helps concretize the proposed method.

### Summary of Weaknesses

1.  **Limited Analysis of the Planning Component:** A significant weakness is the lack of evaluation or error analysis of the graph construction itself. The paper assumes the planned dependency graph is correct and focuses on the execution. How often does the model misidentify dependencies? What happens when it creates an incorrect graph (e.g., misses a dependency, creating a cycle)? An analysis of planning accuracy and its impact on final task success is a critical missing piece.

2.  **Oversimplified Reward Function:** The RL reward function (`R_acc(τ) = score_answer`) is extremely simplistic, relying on a binary correctness signal. This is a missed opportunity. The authors could have designed a richer reward function that also incentivizes efficiency (e.g., penalizing excessive turns or token length) or planning correctness, which might lead to even stronger results. The justification that "format consistency is inherently ensured" is not fully convincing without supporting evidence.

3.  **Ambiguity in Parallel Execution:** The paper is somewhat vague on the practical implementation of "parallel tool calling." Does the system spawn multiple concurrent threads/processes? How are the results from parallel tool calls aggregated and presented to the LLM? A more detailed description of the execution engine would be beneficial for reproducibility and to understand the source of the latency improvements.

4.  **Incremental Performance Gains:** While the efficiency gains are impressive and clearly significant, the raw accuracy improvements over the strongest baselines (like AFM-RL-3B) are relatively modest (e.g., +1.4% on 2Wiki, +0.9% average on multi-hop). The paper would benefit from a discussion that more strongly positions these results: the primary contribution is a paradigm shift towards *efficient* agentic reasoning, where the accuracy gains are a welcome bonus achieved *alongside* major reductions in cost and latency.

### Overall Assessment

This is a **strong and compelling paper** that introduces a genuinely novel and impactful paradigm for tool-using AI agents. Its primary strength lies in its well-executed core idea and its thorough demonstration of significant efficiency gains. The weaknesses are primarily in the analysis of the method's inner workings (planning accuracy) and some implementation details. The clarity of the presentation is generally high, making the conceptual advance easy to grasp. The work represents a meaningful step forward from purely sequential reasoning and has clear practical implications for deploying more capable and cost-effective AI agents.

---

# Communication and Verification in LLM Agents towards Collaboration under Information Asymmetry

Authors: Run Peng, Ziqiao Ma, Amy Pang, Sikai Li, Zhang Xi-Jia, Yingzhuo Yu, Cristian-Paul Bara, Joyce Chai

Keywords: LLM agents, multi-agent collaboration, information asymmetry, communication strategies, environment-based verification, human-AI interaction, constraint satisfaction, reasoning verification

Comments: Workshop on Multi-Agent System @ ICML 2025

Paper link: [http://arxiv.org/abs/2510.25595v1](http://arxiv.org/abs/2510.25595v1)

## Abstract

While Large Language Model (LLM) agents are often approached from the angle of action planning/generation to accomplish a goal (e.g., given by language descriptions), their abilities to collaborate with each other to achieve a joint goal are not well explored. To address this limitation, this paper studies LLM agents in task collaboration, particularly under the condition of information asymmetry, where agents have disparities in their knowledge and skills and need to work together to complete a shared task. We extend Einstein Puzzles, a classical symbolic puzzle, to a table-top game. In this game, two LLM agents must reason, communicate, and act to satisfy spatial and relational constraints required to solve the puzzle. We apply a fine-tuning-plus-verifier framework in which LLM agents are equipped with various communication strategies and verification signals from the environment. Empirical results highlight the critical importance of aligned communication, especially when agents possess both information-seeking and -providing capabilities. Interestingly, agents without communication can still achieve high task performance; however, further analysis reveals a lack of true rule understanding and lower trust from human evaluators. Instead, by integrating an environment-based verifier, we enhance agents' ability to comprehend task rules and complete tasks, promoting both safer and more interpretable collaboration in AI systems. https://github.com/Roihn/EinsteinPuzzles

## Summary

This paper investigates collaboration between LLM agents under conditions of information asymmetry, where agents have different pieces of knowledge and must work together to solve a shared task. The authors adapt Einstein's Puzzle into a tabletop game where two agents must place objects in target bins based on spatial and relational constraints, with each agent possessing only partial information about the constraints. The key contributions include exploring different communication strategies and introducing an environment-based verifier to enhance collaboration.

The method involves a fine-tuning-plus-verifier framework where LLM agents are equipped with varying communicative action spaces: information providing only, information seeking only, both providing and seeking, or no communication. The authors fine-tuned open-source models (Llama and Qwen variants) and evaluated them in self-play scenarios. A novel environment-based verifier was introduced, which uses environmental feedback and constraint reasoning to validate agent actions without additional training, improving decision-making through trial-and-error.

Key results show that agents with both information seeking and providing capabilities achieve the highest task success rates, while mismatched communication abilities between agents lead to significant performance degradation. The environment-based verifier substantially improved success rates across all configurations (e.g., +30.66% for Llama3.1-8B with CoT in the full communication setting). Error analysis revealed that the verifier reduces errors in rule understanding and redundant communication. A human study indicated that while agents without communication can be efficient in self-play, humans prefer proactive communicators for better clarity and trust, highlighting a gap between task efficiency and human preference in collaborative settings.

## Critique

Of course. Here is a critique of the paper "Communication and Verification in LLM Agents towards Collaboration under Information Asymmetry."

### Overall Assessment

This is a strong, well-executed paper that makes a clear and valuable contribution to the field of multi-agent LLM systems. It addresses an important and underexplored problem—collaboration under information asymmetry—with a rigorous experimental approach and insightful findings.

---

### Strengths

1.  **Novel and Well-Defined Problem Setting:** The core contribution is the focus on *symmetric* collaboration under *information asymmetry*. This is a more challenging and realistic scenario than many prior works that rely on information transparency or asymmetric roles (e.g., manager/worker). The adaptation of Einstein's Puzzle into a structured tabletop game provides a clean, controlled, yet non-trivial testbed for this problem.

2.  **Systematic Experimental Design:** The paper's methodology is a major strength. The authors systematically dissect the problem by defining four distinct communicative action spaces (Provide & Seek, Provide Only, Seek Only, None). This allows for a granular understanding of how different communication capabilities impact collaboration, moving beyond a simple "communication vs. no communication" comparison.

3.  **Introduction of a Lightweight, Practical Solution (Environment-based Verifier):** The proposed "environment-based verifier" is a clever and highly practical innovation. Instead of training a separate, complex value model, it leverages the inherent structure of the simulated environment to provide feedback. This approach is training-free, computationally efficient, and demonstrates significant performance gains, making it an attractive solution for similar interactive environments.

4.  **Comprehensive and Convincing Results:** The results are extensive and answer the posed research questions clearly.
    *   **RQ1/RQ2:** The hierarchy of communication effectiveness (Provide & Seek > Seek Only > None > Provide Only) is a key finding. The dramatic performance improvement from the verifier, coupled with the detailed error analysis in Table 2, provides strong evidence for its utility in improving both task performance and rule understanding.
    *   **RQ3:** The experiment with mismatched action spaces is a brilliant addition, revealing the critical importance of aligned communication protocols in multi-agent systems, a crucial insight for real-world deployment.
    *   **RQ4:** The human study is the crown jewel of the paper. It reveals the critical gap between task efficiency and human preference. The finding that humans value proactive communication (providing) for clarity and trust, even if it's less "efficient" in a pure step-count metric, is highly significant for human-AI collaboration.

5.  **Clarity and Presentation:** The paper is generally well-written and structured. The figures and tables are clear and effectively support the narrative. The inclusion of a detailed error taxonomy (Table 2) adds substantial depth to the analysis.

---

### Weaknesses

1.  **Limited Generalizability Discussion:** While the paper rightly highlights the verifier's applicability to other "simulated environments," it could do more to discuss the specific *types* of environments where this approach is most suitable. The verifier relies on a well-defined, symbolic state and deterministic rules. Its transferability to more ambiguous, continuous, or partially observable environments is less clear and could be discussed as a limitation.

2.  **Scale of Human Study:** The authors correctly identify this as a limitation. With only 12 participants, the human preference results, while insightful and well-aligned with the quantitative data, would benefit from a larger-scale study to strengthen their statistical significance and generalizability.

3.  **Potential Overfitting in "No Communication" Mode:** The surprisingly high performance of some models in the "No Information Exchange" condition is intriguing but slightly worrying. The authors attribute it to models memorizing "high-probability transition patterns." This suggests a potential brittleness or overfitting to the specific game distribution, which could be explored further. It reinforces their main point that high success rates without communication do not imply true understanding.

4.  **Clarity on Verifier Implementation:** The description of the "graph expansion algorithm" within the reasoning verifier (Section 4) is somewhat brief. A more detailed explanation or a small pseudocode snippet in the appendix would help readers fully grasp this component, which is central to the method's success.

---

### Conclusion

This is an excellent paper that makes a meaningful contribution. Its strengths—a novel problem formulation, a systematic experimental design, a pragmatic and effective solution, and insightful results bridging AI performance and human preference—far outweigh its minor weaknesses. The findings on the necessity of aligned communication and the value of environment-based verification will likely influence future research in multi-agent and human-AI collaborative systems.

---

# ProMediate: A Socio-cognitive framework for evaluating proactive agents in multi-party negotiation

Authors: Ziyi Liu, Bahar Sarrafzadeh, Pei Zhou, Longqi Yang, Jieyu Zhao, Ashish Sharma

Keywords: Proactive Agents, Multi-party Negotiation, Socio-cognitive Framework, AI Mediation, Consensus Tracking, Multi-agent Collaboration, Social Intelligence, Evaluation Metrics

Comments: None

Paper link: [http://arxiv.org/abs/2510.25224v1](http://arxiv.org/abs/2510.25224v1)

## Abstract

While Large Language Models (LLMs) are increasingly used in agentic frameworks to assist individual users, there is a growing need for agents that can proactively manage complex, multi-party collaboration. Systematic evaluation methods for such proactive agents remain scarce, limiting progress in developing AI that can effectively support multiple people together. Negotiation offers a demanding testbed for this challenge, requiring socio-cognitive intelligence to navigate conflicting interests between multiple participants and multiple topics and build consensus. Here, we present ProMediate, the first framework for evaluating proactive AI mediator agents in complex, multi-topic, multi-party negotiations. ProMediate consists of two core components: (i) a simulation testbed based on realistic negotiation cases and theory-driven difficulty levels (ProMediate-Easy, ProMediate-Medium, and ProMediate-Hard), with a plug-and-play proactive AI mediator grounded in socio-cognitive mediation theories, capable of flexibly deciding when and how to intervene; and (ii) a socio-cognitive evaluation framework with a new suite of metrics to measure consensus changes, intervention latency, mediator effectiveness, and intelligence. Together, these components establish a systematic framework for assessing the socio-cognitive intelligence of proactive AI agents in multi-party settings. Our results show that a socially intelligent mediator agent outperforms a generic baseline, via faster, better-targeted interventions. In the ProMediate-Hard setting, our social mediator increases consensus change by 3.6 percentage points compared to the generic baseline (10.65\% vs 7.01\%) while being 77\% faster in response (15.98s vs. 3.71s). In conclusion, ProMediate provides a rigorous, theory-grounded testbed to advance the development of proactive, socially intelligent agents.

## Summary

Of course. Here is a summary of the paper "ProMediate: A Socio-cognitive framework for evaluating proactive agents in multi-party negotiation," focusing on its key contributions, methods, and results.

### Summary

**ProMediate** is a novel framework designed to address the lack of systematic evaluation methods for proactive AI agents in complex, multi-party collaborative settings. The paper argues that while AI agents are increasingly used to assist individuals, there is a growing need for agents that can manage group dynamics, such as negotiations, which require socio-cognitive intelligence to navigate conflicting interests and build consensus.

**Key Contributions:**
1.  **An Extensible Testbed:** ProMediate provides a simulation environment built on realistic negotiation cases from Harvard Law School. It features multi-party, multi-topic scenarios with configurable difficulty levels (Easy, Medium, Hard) based on conflict modes (e.g., Accommodating, Avoiding, Competing). The framework includes a plug-and-play architecture for integrating and evaluating different mediator agents.
2.  **A Socio-Cognitive Evaluation Framework:** The paper introduces a comprehensive suite of metrics grounded in mediation theory. These metrics evaluate both conversation-level outcomes (e.g., Consensus Change, Topic-Level Efficiency) and mediator-level effectiveness (e.g., Response Latency, Mediator Effectiveness, and Mediator Intelligence across perceptual, emotional, cognitive, and communicative dimensions).
3.  **Rigorous Evaluation and Analysis:** The framework is used to compare a "Generic Mediator" with a "Socially Intelligent Mediator" that uses theory-based reasoning to decide when and how to intervene.

**Methods:**
The methodology involves simulating negotiations between LLM-powered human agents. A proactive AI mediator observes the conversation and decides autonomously when to intervene. The "Socially Intelligent Mediator" specifically analyzes the dialogue for breakdowns across four socio-cognitive dimensions and selects from a set of mediation strategies (e.g., Facilitative, Evaluative) to generate its response. The core innovation in evaluation is "consensus tracking," a dynamic method that uses an LLM-as-a-judge to extract participant attitudes and compute a soft, time-varying measure of group agreement throughout the conversation.

**Key Results:**
*   **Agent Effectiveness:** The Socially Intelligent Mediator outperformed the Generic baseline in the most challenging ("Hard"/Competing) setting, achieving a **3.6 percentage point higher consensus change** (10.65% vs. 7.01%) while responding **77% faster**.
*   **Context-Dependent Performance:** The effectiveness of proactivity was context-dependent. In "Easy" settings, frequent interventions could disrupt organic consensus, while in "Hard" settings, they were crucial for breaking deadlocks.
*   **Model Comparison:** Among the models tested (GPT-4.1, Claude-Sonnet-4, o4-mini), the "thinking" model o4-mini performed best in achieving consensus, suggesting that deliberate reasoning leads to higher-quality interventions, even at the cost of slightly higher latency.
*   **Metric Validity:** Factor analysis revealed that the metrics capture two main latent dimensions: "Consensus & Topic Efficiency" and "Intervention Dynamics/Tempo." A critical finding was that high "Mediator Intelligence" scores did not guarantee immediate "Mediator Effectiveness," highlighting that smart interventions can sometimes surface disagreements for long-term gain at the expense of short-term consensus.

In conclusion, ProMediate establishes a rigorous, theory-grounded testbed for developing and evaluating proactive, socially intelligent AI agents capable of facilitating complex multi-party interactions.

## Critique

Of course. Here is a critique of the paper "ProMediate: A Socio-cognitive framework for evaluating proactive agents in multi-party negotiation," covering its strengths, weaknesses, and overall clarity.

### Strengths

1.  **High Novelty and Timely Contribution:** The paper addresses a significant and underexplored gap in AI research: the systematic evaluation of *proactive* AI agents in *complex, multi-party* settings. While there is extensive work on reactive agents and multi-agent systems, the focus on an agent that must decide *when* and *how* to intervene in a human-like group dynamic is a substantial and novel contribution. The framing around "socio-cognitive intelligence" provides a strong theoretical foundation.

2.  **Comprehensive and Rigorous Framework:** ProMediate is not just a dataset; it's a full-stack evaluation framework. Its two core components—a realistic simulation testbed with configurable difficulty levels and a multi-faceted evaluation suite—are well-integrated and thoughtfully designed. The use of real-world negotiation cases from Harvard Law School adds ecological validity.

3.  **Sophisticated, Multi-Dimensional Evaluation:** The proposed metrics are a major strength. Moving beyond a single final-score metric to a suite that captures dynamic consensus tracking (CC, TLE), intervention timing (RL), immediate impact (ME), and underlying intelligence (MI) provides a much richer and more nuanced understanding of agent performance. The factor analysis revealing the two latent factors (*Consensus & Topic Efficiency* and *Intervention Dynamics / Tempo*) is a compelling validation of this multi-dimensional approach.

4.  **Insightful and Nuanced Results:** The results are significant because they are not simply "AI improves everything." The key finding—that a socially intelligent mediator excels in hard settings but can be disruptive in easy ones—is crucial. It demonstrates that proactivity is a double-edged sword and that context-aware intervention strategies are essential. The finding that high Mediator Intelligence (MI) does not guarantee immediate consensus gains (ME) is a sophisticated and important insight, highlighting the complex, non-deterministic nature of human (and simulated human) interaction.

### Weaknesses

1.  **Limited Scope of "Social Intelligence":** While the socio-cognitive framework (perceptual, emotional, cognitive, communicative) is a good start, the implementation of the "Socially Intelligent Mediator" is primarily based on a reasoning process and strategy selection. It does not convincingly demonstrate deeper social capabilities like long-term relationship building, nuanced emotional empathy, or adapting a unique "persona" as a mediator. The performance gap between the "Social" and "Generic" agents, while present, is not overwhelming, suggesting there is significant room for improvement in how social intelligence is operationalized.

2.  **Heavy Reliance on LLM-as-a-Judge:** The entire evaluation pipeline—from attitude extraction and agreement scoring to mediator intelligence scoring—relies on a proprietary LLM (GPT-4.1). This introduces potential biases, high costs, and reproducibility concerns as the model version changes. The paper would be stronger if it included ablations with other models or proposed more automated, model-agnostic metrics for at least some of the evaluations (e.g., using semantic similarity or keyword analysis as a baseline for consensus).

3.  **Scalability and Computational Cost:** The simulation framework, which involves multiple LLMs running complex reasoning chains for each turn, is described as taking "1 to 3 hours" per conversation. This limits the scale of experimentation (only 5 runs per scenario) and makes the framework computationally prohibitive for many research groups, potentially hindering widespread adoption.

4.  **Validation on Simulated Humans:** The entire experiment is conducted with simulated human agents (powered by Claude-Sonnet-4). While the human evaluation confirms the conversations are "natural," the ultimate test of a mediator's effectiveness is with real human participants. The dynamics of persuasion, resistance, and emotional response in real humans may differ significantly from even the best simulators, and this remains an unvalidated leap.

### Clarity of Presentation

The paper is generally well-written and clearly structured. The use of research questions (RQ1-3) effectively guides the reader through the results and analysis. The figures and tables are relevant and support the narrative.

**Areas for Improvement in Clarity:**
*   The description of the "InnerThought framework" in Section 2.2 is somewhat brief, and the reader is referred to an appendix for details. A one-paragraph summary in the main text would improve flow.
*   The distinction between the four mediation strategies (Facilitative, Evaluative, etc.) is relegated to the appendix. A concise definition of each in the main text (Section 4.1) would help the reader better understand the "How" of the social mediator.
*   The correlation scatterplot between ME and MI mentioned in Section 4.3.3 is cited as being in the appendix (Figure 4); including it in the main text would strengthen the argument.

### Overall Assessment

This is a strong, high-quality paper that makes a valuable contribution to the fields of AI, multi-agent systems, and human-computer interaction. Its primary strength lies in its novel and comprehensive framework for a critically important yet understudied problem. The nuanced results provide genuine scientific insight rather than just a performance leaderboard. The main weaknesses relate to the inherent limitations of current simulation-based evaluation (cost, scalability, and the sim-to-real gap) and the operationalization of "social intelligence." Despite these, ProMediate successfully establishes a rigorous, theory-grounded benchmark that will likely inspire and enable significant future research.

---

# DEBATE: A Large-Scale Benchmark for Role-Playing LLM Agents in Multi-Agent, Long-Form Debates

Authors: Yun-Shiuan Chuang, Ruixuan Tu, Chengtao Dai, Smit Vasani, Binwei Yao, Michael Henry Tessler, Sijia Yang, Dhavan Shah, Robert Hawkins, Junjie Hu, Timothy T. Rogers

Keywords: multi-agent systems, role-playing LLMs, opinion dynamics, debate simulation, human-agent alignment, social interaction modeling

Comments: None

Paper link: [http://arxiv.org/abs/2510.25110v1](http://arxiv.org/abs/2510.25110v1)

## Abstract

Accurately modeling opinion change through social interactions is crucial for addressing issues like misinformation and polarization. While role-playing large language models (LLMs) offer a promising way to simulate human-like interactions, existing research shows that single-agent alignment does not guarantee authentic multi-agent group dynamics. Current LLM role-play setups often produce unnatural dynamics (e.g., premature convergence), without an empirical benchmark to measure authentic human opinion trajectories. To bridge this gap, we introduce DEBATE, the first large-scale empirical benchmark explicitly designed to evaluate the authenticity of the interaction between multi-agent role-playing LLMs. DEBATE contains 29,417 messages from multi-round debate conversations among over 2,792 U.S.-based participants discussing 107 controversial topics, capturing both publicly-expressed messages and privately-reported opinions. Using DEBATE, we systematically evaluate and identify critical discrepancies between simulated and authentic group dynamics. We further demonstrate DEBATE's utility for aligning LLMs with human behavior through supervised fine-tuning, achieving improvements in surface-level metrics (e.g., ROUGE-L and message length) while highlighting limitations in deeper semantic alignment (e.g., semantic similarity). Our findings highlight both the potential and current limitations of role-playing LLM agents for realistically simulating human-like social dynamics.

## Summary

Based on the provided paper "DEBATE: A Large-Scale Benchmark for Role-Playing LLM Agents in Multi-Agent, Long-Form Debates," here is a concise summary focusing on its key contributions, methods, and results:

### Key Contributions
This paper introduces **DEBATE**, the first large-scale empirical benchmark designed to evaluate the authenticity of multi-agent role-playing LLM systems in simulating human opinion dynamics. The dataset contains **29,417 messages** from **2,792 U.S.-based participants** engaging in multi-round debates across **107 controversial topics**. DEBATE captures both publicly expressed opinions (tweet-like messages) and privately reported beliefs (Likert-scale ratings), enabling comprehensive evaluation at utterance, individual, and group levels. The benchmark supports three simulation modes (Next Message Prediction, Tweet-guided Conversation Simulation, and Full Conversation Simulation) to study different aspects of multi-agent communication.

### Methods
The authors designed a multi-agent conversational experiment where participants in 4-person groups engaged in 3 rounds of dyadic debates on controversial topics. Each round involved writing tweet-like posts and real-time conversations, with pre- and post-discussion opinion measurements. The researchers then constructed role-playing LLM agents as "digital twins" of human participants, conditioning them on demographic profiles, initial opinions, and conversation histories. They evaluated six LLMs across three simulation modes using metrics including semantic similarity, stance difference, message length, ROUGE-L, and on-topic utterance rate. Additionally, they conducted ablation studies to understand the importance of different memory components and explored supervised fine-tuning for behavioral alignment.

### Key Results
The evaluation revealed several important findings: (1) **gpt-4o-mini consistently showed the strongest alignment** with human responses across most metrics; (2) **Alignment performance declined across simulation modes**, with Mode 1 (Next Message Prediction) performing best and Mode 3 (Full Conversation) performing worst; (3) **LLM-simulated groups exhibited systematic behavioral differences** from human groups, including stronger opinion convergence, positive drift in public tweet stance, and more systematic individual shifts (stronger regression to the mean and greater susceptibility to partner influence); (4) **Supervised fine-tuning improved surface-level metrics** (message length, ROUGE-L) but failed to enhance deeper semantic or stance alignment; (5) **Private profile information became increasingly important** in simulation modes where conversation history was recursively generated.

The paper concludes that while current LLM agents can reproduce some utterance-level patterns, they fall short in modeling authentic human opinion dynamics, highlighting the need for more socially grounded multi-agent systems.

## Critique

Here is a critique of the paper "DEBATE: A Large-Scale Benchmark for Role-Playing LLM Agents in Multi-Agent, Long-Form Debates":

**Strengths:**

*   **High Novelty and Significance:** The paper addresses a critical and underexplored gap: the lack of a large-scale, empirical benchmark for evaluating the *emergent group dynamics* of multi-agent, role-playing LLMs. While single-agent role-playing has been studied, this work correctly identifies that realistic individual behavior does not guarantee realistic group interactions. The creation of the DEBATE benchmark is a significant contribution to the field.

*   **Comprehensive and Rigorous Dataset:** The dataset is a major strength. It is large-scale (29,417 messages from 2,792 participants), carefully designed (with depth and breadth topics), and captures a rich, multi-faceted view of opinion dynamics, including both public expressions (tweets, conversations) and private beliefs (Likert-scale ratings). The demographic diversity of participants adds to its robustness.

*   **Well-Structured Evaluation Framework:** The paper introduces a clear and multi-level evaluation strategy (utterance, individual, and group levels) with appropriate metrics (semantic similarity, stance difference, ROUGE-L, etc.). The definition of three distinct simulation modes (Next Message Prediction, Tweet-guided, Full Conversation) is insightful, as it allows for a granular analysis of how different levels of human grounding affect performance.

*   **Compelling and Actionable Findings:** The results are significant and go beyond simple performance metrics. The key findings—such as LLM agents exhibiting premature consensus, a positive drift in public stance, and stronger regression-to-the-mean effects compared to humans—provide concrete evidence of the limitations of current multi-agent LLM systems. The ablation studies and the preliminary SFT experiments offer valuable insights into what information is important for agents and the challenges of improving alignment through simple fine-tuning.

*   **Clarity of Presentation:** The paper is generally well-written and logically structured. The use of figures and tables effectively summarizes complex results. The methodology for agent construction and simulation is described in sufficient detail.

**Weaknesses:**

*   **Limited Model Scope:** The evaluation, while covering several prominent open-weight models, is heavily centered on the GPT-4o-mini API. A broader analysis including the full-scale GPT-4o, GPT-4, or other top-tier proprietary models would have provided a more complete picture of the state-of-the-art. The conclusion that GPT-4o-mini is the best might be true within the tested set, but its absolute performance level remains an open question.

*   **Superficial Treatment of Fine-Tuning:** The SFT experiments are mentioned as a preliminary finding in an appendix. Given the importance of alignment methods, a more thorough investigation of SFT and other techniques like Reinforcement Learning from Human Feedback (RLHF) or Direct Preference Optimization (DPO) would have been a valuable addition to the main paper, even if results were mixed.

*   **Unexplored Mechanisms for Observed Biases:** The paper excellently documents *that* LLM agents converge more and drift positively, but the analysis of *why* this happens is somewhat surface-level. A deeper discussion or hypothesis about the root causes—e.g., is it a bias in the base model's training data, an effect of the "helpful and harmless" alignment, or an artifact of the prompting strategy?—would have strengthened the paper's impact.

*   **Clarity of the "Positive Drift" Finding:** The finding that LLM agents' public stance becomes more positive on topics with a known *false* ground truth is intriguing but could be better contextualized. Is this a general positivity bias, or is it specific to the model's latent "knowledge" conflicting with the assigned persona? A more detailed interpretation of this specific result would be helpful.

**Overall Assessment:**

This is a strong and highly valuable paper. Its primary contribution is the creation and validation of the DEBATE benchmark itself, which is likely to become a standard tool for evaluating multi-agent LLM systems. The empirical findings are robust, significant, and clearly demonstrate that current models fail to capture the nuanced complexity of human group dynamics. While there are opportunities for deeper analysis, particularly regarding the underlying causes of the observed biases and a broader model evaluation, the work presented is a substantial step forward for the field. The paper is well-argued, the experiments are thorough, and the results are compelling.

