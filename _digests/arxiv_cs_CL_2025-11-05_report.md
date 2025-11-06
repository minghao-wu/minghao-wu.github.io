---
title: "ArXiv Daily Digest on 2025-11-05"
collection: digests
permalink: /digests/arxiv_cs_CL_2025-11-05_report
date: 2025-11-05
location: "Online"
---

Today's research highlights significant advances in enhancing computational efficiency and security for large language models (LLMs), with two key themes emerging: innovative architectural scaling for continual learning and sophisticated adversarial evaluation frameworks. In continual learning, the SCALE (Upscaled Continual Learning) architecture demonstrates that width upscaling with strategic parameter freezing can dramatically mitigate catastrophic forgetting in small language models (SLMs), enabling effective knowledge acquisition while preserving original capabilities. Concurrently, the EQ-Negotiator framework shows how dynamic emotional personas can empower SLMs to match or exceed LLM performance in complex tasks like credit negotiation, emphasizing strategic intelligence over model scale. Meanwhile, in evaluation methodologies, studies reveal how source-aware neural machine translation (MT) metrics can be reliably adapted for speech translation (ST) using synthetic sources, and novel attack frameworks like Dynamic Deceptor (DyDec) and Static Deceptor (StaDec) expose critical vulnerabilities in LLMs through adaptive, transferable adversarial examples, underscoring the urgent need for improved robustness measures.

## TL;DR

Total papers: 46 , Selected papers: 4

### TL;DR: Recent Advances in Language Model Applications & Security

This week's papers focus on enhancing language models' capabilities while addressing their vulnerabilities:

**🧠 Multi-Agent & Emotional Intelligence**
- **EQ-Negotiator** enables small language models to match LLM performance in credit negotiations through dynamic emotional personas and strategic reasoning (HMM + game theory), making edge deployment feasible.  
  *https://arxiv.org/abs/2511.03370v1*

**🔄 Continual Learning Architectures**  
- **SCALE** introduces width upscaling with frozen base parameters for continual pre-training, significantly reducing catastrophic forgetting while maintaining plasticity.  
  *https://arxiv.org/abs/2511.03270v1*

**🎯 Speech Translation Evaluation**
- Systematic study shows source-aware MT metrics can be effectively adapted for speech translation using ASR transcripts or back-translations, with a novel cross-lingual segmentation algorithm.  
  *https://arxiv.org/abs/2511.03295v1*

**⚔️ Adversarial Security**
- **Dynamic/Static Deceptor** frameworks use LLM collaboration to generate adaptive adversarial examples that bypass current defenses and show strong transferability across models.  
  *https://arxiv.org/abs/2511.03128v1*

**Key Insights**: Strategic architectures (emotional reasoning, width scaling) can compensate for model size; evaluation methodologies need adaptation for multimodal inputs; and LLM collaboration creates both powerful applications and new security challenges.

---

# EQ-Negotiator: Dynamic Emotional Personas Empower Small Language Models for Edge-Deployable Credit Negotiation

Authors: Yunbo Long, Yuhan Liu, Alexandra Brintrup

Keywords: Emotional Intelligence, Credit Negotiation, Small Language Models, Edge Deployment, Multi-Agent Systems, Hidden Markov Model

Comments: None

Paper link: [http://arxiv.org/abs/2511.03370v1](http://arxiv.org/abs/2511.03370v1)

## Abstract

The deployment of large language models (LLMs) in automated negotiation has set a high performance benchmark, but their computational cost and data privacy requirements render them unsuitable for many privacy-sensitive, on-device applications such as mobile assistants, embodied AI agents or private client interactions. While small language models (SLMs) offer a practical alternative, they suffer from a significant performance gap compared to LLMs in playing emotionally charged complex personas, especially for credit negotiation. This paper introduces EQ-Negotiator, a novel framework that bridges this capability gap using emotional personas. Its core is a reasoning system that integrates game theory with a Hidden Markov Model(HMM) to learn and track debtor emotional states online, without pre-training. This allows EQ-Negotiator to equip SLMs with the strategic intelligence to counter manipulation while de-escalating conflict and upholding ethical standards. Through extensive agent-to-agent simulations across diverse credit negotiation scenarios, including adversarial debtor strategies like cheating, threatening, and playing the victim, we show that a 7B parameter language model with EQ-Negotiator achieves better debt recovery and negotiation efficiency than baseline LLMs more than 10 times its size. This work advances persona modeling from descriptive character profiles to dynamic emotional architectures that operate within privacy constraints. Besides, this paper establishes that strategic emotional intelligence, not raw model scale, is the critical factor for success in automated negotiation, paving the way for effective, ethical, and privacy-preserving AI negotiators that can operate on the edge.

## Summary

Based on the provided paper, here is a summary of its key contributions, methods, and results.

**Title:** EQ-Negotiator: Dynamic Emotional Personas Empower Small Language Models for Edge-Deployable Credit Negotiation

**Key Contributions:**
This paper introduces EQ-Negotiator, a novel framework designed to bridge the performance gap between large language models (LLMs) and small language models (SLMs) in automated negotiation tasks. Its primary contribution is enabling SLMs to operate with sophisticated emotional intelligence on edge devices, addressing critical limitations of cloud-based LLMs such as data privacy risks, latency, and high computational costs. The work establishes that strategic emotional intelligence, rather than raw model scale, is a decisive factor for success in complex, emotionally charged interactions like credit negotiation.

**Methods:**
The core of the EQ-Negotiator framework is a reasoning system that dynamically adapts a creditor agent's emotional responses during negotiations with a debtor agent. The methodology integrates several key components:
1.  **In-Context Emotion Recognition:** The system classifies the debtor's emotional state in real-time from the dialogue using in-context learning with a seven-emotion framework (Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral), eliminating the need for fine-tuning.
2.  **Dynamic Emotional Shift Strategies:** It employs two main strategies for selecting the creditor's emotional response:
    *   **Win-Stay, Lose-Shift (WSLS):** A game-theoretic approach that maintains a cooperative emotional stance following positive debtor emotions but shifts strategy after negative interactions.
    *   **Hidden Markov Model (HMM):** Activated when a sequence of negative emotions is detected, this model infers the debtor's hidden strategic state (e.g., Cooperative, Confrontational) to predict and select optimal emotional responses.
3.  **Multi-Agent Simulation:** The framework is evaluated through automated simulations involving a debtor agent, an EQ-Negotiator-equipped creditor agent, and an independent judge agent that determines negotiation outcomes.

**Results:**
The framework was evaluated on a credit negotiation dataset against various debtor personas, including emotionally charged and adversarial strategies (e.g., cheating, threatening).
*   **Performance Enhancement:** EQ-Negotiator significantly boosted the performance of SLMs (7B parameters). For instance, Llama-7B's success rate increased from 40% to 70%, and it achieved superior debt collection terms compared to vanilla (non-EQ) LLMs in some adversarial scenarios.
*   **Competitiveness with LLMs:** An SLM equipped with EQ-Negotiator dramatically narrowed the performance gap with much larger LLMs (e.g., GPT-4o-mini), demonstrating that emotional intelligence can compensate for a lack of model scale.
*   **Ethical Improvement:** The framework also reduced unethical negotiation behaviors (e.g., manipulative language, false empathy) in both SLMs and LLMs, with the most substantial improvements observed in SLMs.

In conclusion, EQ-Negotiator provides a practical and effective pathway for deploying privacy-preserving, emotionally intelligent negotiation agents on resource-constrained edge devices.

## Critique

Of course. Here is a critique of the paper "EQ-Negotiator: Dynamic Emotional Personas Empower Small Language Models for Edge-Deployable Credit Negotiation," focusing on its strengths, weaknesses, novelty, and clarity.

### Overall Summary

This paper presents a compelling framework that enhances the emotional intelligence of Small Language Models (SLMs) for automated credit negotiation, enabling them to perform on par with or even surpass much larger LLMs in specific scenarios. The work is timely, addressing critical concerns of privacy, cost, and latency for edge deployment.

---

### Strengths

1.  **High Practical Relevance and Problem Formulation:** The paper effectively identifies a significant and growing problem: the unsuitability of cloud-based LLMs for privacy-sensitive, real-time negotiation tasks (e.g., on mobile devices, in embodied AI). The focus on credit negotiation provides a concrete, high-stakes domain where emotional dynamics are crucial.
2.  **Novel and Well-Motivated Approach:** The core idea of using a hybrid game theory (Win-Stay, Lose-Shift) and Hidden Markov Model (HMM) framework to guide an SLM's emotional responses is innovative. It cleverly compensates for the SLM's lack of inherent nuanced reasoning by providing it with an external, strategic "emotional engine." The shift from static persona profiles to dynamic emotional architectures is a notable conceptual contribution.
3.  **Comprehensive and Rigorous Evaluation:** The experimental design is thorough. It tests the framework against:
    *   **Baseline Models:** Comparing SLMs and LLMs with and without the EQ-Negotiator.
    *   **Diverse Adversarial Scenarios:** Including fixed negative emotions and sophisticated manipulation tactics (cheating, threatening, stonewalling).
    *   **Multiple Metrics:** Success rate, financial outcome (debt collection multiples), and efficiency (negotiation speed).
    *   **Ethical Analysis:** A dedicated evaluation of manipulative behavior is a significant strength, adding a crucial dimension often missing from AI agent research.
4.  **Significant and Surprising Results:** The results are the paper's strongest asset. The finding that a 7B model equipped with EQ-Negotiator can outperform vanilla LLMs "more than 10 times its size" is a powerful demonstration of the framework's value. The observation of "model homophily" (identical models negotiating more effectively) is an interesting and valuable insight. The ethical analysis convincingly shows that the framework not only improves performance but also makes SLMs more ethical negotiators.
5.  **Excellent Clarity and Structure:** The paper is very well-written and structured. The pipeline diagram (Figure 1) provides a clear overview, and the use of algorithms in the main text and appendix makes the methodology concrete and reproducible. The presentation of results with confidence intervals is professionally done.

---

### Weaknesses

1.  **Limited Explainability of the Core Mechanism:** While the HMM's function is well-described mathematically, there is limited analysis of *why* it makes specific emotional choices in practice. A qualitative analysis of a few negotiation transcripts, showing how the HMM's state transitions led to a successful counter-strategy, would greatly strengthen the paper and make the "reasoning" less of a black box.
2.  **Simplified and Constrained Emotional Model:** The restriction to only seven basic, discrete emotions (based on Ekman's model) is a significant limitation. Real-world negotiations involve complex, blended, and culturally nuanced emotional states. The paper acknowledges this, but it remains a fundamental constraint on the model's realism.
3.  **Uncertain Generalizability:** The framework is extensively validated only in the domain of credit negotiation. It is unclear how well it would transfer to other domains like diplomatic negotiations, business deals, or everyday customer service, where the payoff matrices, emotional dynamics, and ethical constraints are vastly different. The "cross-cultural" limitation mentioned is particularly pertinent.
4.  **Dependence on High-Quality Prompts:** The performance of the entire system is heavily reliant on the carefully crafted prompts for emotion detection and strategy implementation (detailed in the appendix). The paper does not explore the sensitivity of the results to variations in these prompts, which is a potential fragility.
5.  **Synthetic Evaluation Environment:** While multi-agent simulation is a valid and scalable evaluation method, all results are generated in a simulated environment with AI debtors. The ultimate test would be performance against human negotiators, who exhibit much more unpredictable and complex emotional behaviors.

---

### Assessment of Novelty, Significance, and Clarity

*   **Novelty:** **High.** The paper's primary novelty lies in its hybrid架构 (HMM + Game Theory) for dynamic emotional intelligence in SLMs, specifically tailored for edge deployment. Moving beyond fine-tuning to an online, reasoning-based guidance system is a distinct approach from most prior work.
*   **Significance:** **High.** The work has significant implications for the field. It provides a practical blueprint for creating capable, private, and cost-effective AI agents for real-world applications. It challenges the prevailing narrative that model scale is the primary path to performance, demonstrating that strategic architecture can be a powerful alternative.
*   **Clarity:** **Excellent.** The paper is exceptionally clear. The logical flow from introduction to conclusion is smooth, the methodology is well-explained with formulas and algorithms, and the results are presented comprehensively with strong visualizations.

### Conclusion

This is a strong, well-executed paper that makes a valuable contribution to the fields of AI negotiation, language agents, and edge AI. Its strengths in problem formulation, novel methodology, and compelling results far outweigh its weaknesses, which are common challenges in the field (e.g., explainability, generalizability). The work successfully demonstrates that strategic emotional intelligence can be a decisive factor, effectively bridging the performance gap between SLMs and LLMs for a critical class of applications.

---

# SCALE: Upscaled Continual Learning of Large Language Models

Authors: Jin-woo Lee, Junhwa Choi, Bongkyu Hwang, Jinho Choo, Bogun Kim, JeongSeon Yi, Joonseok Lee, DongYoung Jung, Jaeseon Park, Kyoungwon Park, Suk-hoon Jung

Keywords: Continual Learning, Large Language Models, Width Upscaling, Persistent Preservation, Collaborative Adaptation, SCALE Architecture, Catastrophic Forgetting, Stability-Plasticity Trade-off

Comments: None

Paper link: [http://arxiv.org/abs/2511.03270v1](http://arxiv.org/abs/2511.03270v1)

## Abstract

We revisit continual pre-training for large language models and argue that progress now depends more on scaling the right structure than on scaling parameters alone. We introduce SCALE, a width upscaling architecture that inserts lightweight expansion into linear modules while freezing all pre-trained parameters. This preserves the residual and attention topologies and increases capacity without perturbing the base model's original functionality. SCALE is guided by two principles: Persistent Preservation, which maintains the base model's behavior via preservation-oriented initialization and freezing of the pre-trained weights, and Collaborative Adaptation, which selectively trains a subset of expansion components to acquire new knowledge with minimal interference. We instantiate these ideas as SCALE-Preserve (preservation-first), SCALE-Adapt (adaptation-first), and SCALE-Route, an optional routing extension that performs token-level routing between preservation and adaptation heads. On a controlled synthetic biography benchmark, SCALE mitigates the severe forgetting observed with depth expansion while still acquiring new knowledge. In continual pre-training on a Korean corpus, SCALE variants achieve less forgetting on English evaluations and competitive gains on Korean benchmarks, with these variants offering the best overall stability-plasticity trade-off. Accompanying analysis clarifies when preservation provably holds and why the interplay between preservation and adaptation stabilizes optimization compared to standard continual learning setups.

## Summary

Based on the provided paper, here is a concise summary focusing on its key contributions, methods, and results.

**Summary**

The paper introduces **SCALE**, a novel architectural framework for **upscaled continual learning (CL)** in large language models (LLMs). The core argument is that future progress in LLMs depends less on simply scaling parameters and more on strategically scaling the model's *structure* to preserve pre-existing knowledge while acquiring new information. SCALE addresses the issue of catastrophic forgetting by performing **width upscaling**—expanding the hidden dimensions inside linear modules—while **freezing all pre-trained parameters**. This approach preserves the original model's computational graph and functionality.

**Key Contributions & Methods**

1.  **Width Upscaling Architecture:** SCALE freezes the base model's weights and increases capacity by adding lightweight expansion blocks to linear layers (in MHA and FFN modules). This is represented by upscaling the weight matrix `W` into blocks `[W, W12; W21, W22]`, where the new blocks introduce additional parameters.

2.  **Two Core Design Principles:**
    *   **Persistent Preservation:** To maintain the base model's original function, the `W12` blocks are **zero-initialized and frozen**. The paper provides theoretical proof that this ensures the output for the original input dimensions remains unchanged.
    *   **Collaborative Adaptation:** To enable learning of new knowledge, a subset of the upscaled components (e.g., `W12` in upper layers) can be selectively trained. This creates a trade-off between stability (forgetting) and plasticity (learning).

3.  **Three Learning Methods:** Based on these principles, the authors propose three variants:
    *   **SCALE-Preserve:** A preservation-first method where all `W12` blocks are frozen.
    *   **SCALE-Adapt:** An adaptation-first method where all `W12` blocks are trainable.
    *   **SCALE-Route:** A hybrid method that uses a cosine-similarity-based router to dynamically direct tokens through either a preservation or adaptation path during a single forward pass, aiming to get the best of both worlds.

**Key Results**

*   **Controlled Biography Task:** On a synthetic benchmark designed to test catastrophic forgetting, SCALE-Route demonstrated significantly superior knowledge retention. It maintained 100% accuracy on the original task (Task 0) for much longer and finished with a final accuracy of 36.9%, vastly outperforming full fine-tuning (FFT) and depth-upscaling (LLaMA Pro), which dropped to ~15%.
*   **Continual Pre-training on Korean Data:** When continually pre-training LLaMA-3.2-1B on a Korean corpus, all SCALE variants exhibited **significantly less forgetting** on English evaluation data (lower perplexity) compared to FFT, LoRA, and LLaMA Pro. Simultaneously, they achieved **competitive learning performance** on the target Korean data.
*   **Benchmark Evaluation:** On English benchmarks (ARC, HellaSwag, MMLU, etc.), SCALE methods preserved the base model's capabilities much better than other CL methods. While their improvements on Korean benchmarks were more modest, SCALE-Route consistently achieved the best overall **stability-plasticity trade-off** among the evaluated methods.

In conclusion, SCALE presents a practical and effective architectural solution for continual learning in LLMs, demonstrating that width upscaling with strategic freezing and selective training can effectively mitigate catastrophic forgetting while accommodating new knowledge.

## Critique

Of course. Here is a critique of the paper "SCALE: Upscaled Continual Learning of Large Language Models," focusing on its strengths, weaknesses, novelty, and clarity.

### Overall Assessment

This is a strong, well-structured paper that makes a compelling case for width upscaling as a superior alternative to depth upscaling for continual pre-training (CPT) of LLMs. The work is timely, addresses a critical problem (catastrophic forgetting), and is supported by a clear theoretical framework and extensive empirical validation.

---

### Strengths

1.  **Novelty and Clear Positioning:** The paper's core contribution is its sharp focus on **width upscaling** as a more stable alternative to the more common depth upscaling (e.g., LLaMA Pro). The argument that progress now depends on "scaling the right structure" is a compelling and well-motivated premise. The introduction of a family of methods (Preserve, Adapt, Route) based on two design principles provides a nuanced approach to the stability-plasticity dilemma.

2.  **Strong Theoretical Foundation:** The paper goes beyond a purely empirical contribution. The formalization of **"Persistent Preservation"** and **"Collaborative Adaptation"** as design principles is a significant strength. The theoretical analyses in the appendices (function preservation, forgetting bounds, convergence analysis) provide mathematical rigor and explain *why* the proposed method works, which is often missing in architecture-focused ML papers.

3.  **Comprehensive and Well-Designed Experiments:** The evaluation is thorough:
    *   **Controlled Synthetic Task (Biography):** This experiment provides a clean, interpretable demonstration of SCALE's primary advantage: mitigating catastrophic forgetting. The results are stark and convincing, showing SCALE-Route maintaining near-perfect performance on Task 0 far longer than baselines.
    *   **Real-World Continual Pre-training (Korean Corpus):** This validates the method on a practical, large-scale task. The results show SCALE's effectiveness in preserving original (English) capabilities while adapting to a new language domain, demonstrating its real-world applicability.
    *   **Ablation Studies:** The paper includes preliminary studies (e.g., on initialization strategies for \( \mathbf{W}^{21} \) and \( \mathbf{W}^{22} \), the effect of \( L_{fp} \)) that justify its design choices and illustrate the trade-offs between preservation and adaptation.

4.  **Clarity of Presentation:** The paper is generally well-written and logically structured. The use of figures (e.g., Figures 4, 6, 8, 9) effectively illustrates the core concepts and results. The distinction between the three SCALE variants is clear, and the high-level intuition is accessible.

---

### Weaknesses and Limitations

1.  **Modest Scale and Scope:** The authors explicitly note this limitation. Experiments are conducted with models up to 1B parameters (Llama-3.2-1B, Pythia-160M). While the results are promising, the scalability of the approach to models of 10B+ parameters remains an open question. The computational and memory overhead of width expansion at that scale could be significant.

2.  **Simplistic Routing Mechanism:** The routing in SCALE-Route, while effective, is relatively simple (a cosine similarity threshold). The threshold \( \tau \) is a static hyperparameter. A more sophisticated, learnable routing mechanism could potentially yield further improvements and is a clear direction for future work.

3.  **Limited Benchmarking on Target Domain:** In the Korean CPT experiment, while SCALE variants excel at preserving English capabilities, their improvement on the Korean benchmarks (KoBEST) is noted as "marginal" compared to FFT and LoRA. This suggests that while SCALE is excellent for preservation, its absolute performance on the *new* task might be slightly behind methods that are allowed to forget. The trade-off is well-managed, but the peak adaptation performance could be a point of criticism.

4.  **Comparison to Parameter-Efficient Fine-Tuning (PEFT):** The paper compares SCALE against FFT, LLaMA Pro, and Freeze. A more direct comparison with a wider range of state-of-the-art PEFT methods (like (IA)³ or Adapters) in a continual learning setting would strengthen the claim of superiority. While LoRA is included, the broader PEFT landscape is a key related area.

---

### Significance of Results

The results are highly significant for the field of continual learning for LLMs. The paper convincingly demonstrates that:
*   **Width upscaling can be fundamentally more stable** than depth upscaling for CPT, a finding that could redirect research efforts in model expansion.
*   It is possible to achieve a **superior stability-plasticity trade-off** through architectural means, reducing reliance on data replay or strong regularization.
*   The **SCALE-Route** method, in particular, offers a practical and powerful way to dynamically balance preservation and adaptation within a single model.

### Conclusion

This paper presents a novel, theoretically-grounded, and empirically-validated architecture for continual learning in LLMs. Its primary strength lies in its clear demonstration that a specific type of structural scaling (width) is more effective for knowledge preservation than another (depth). The weaknesses are primarily related to the scale of validation and the simplicity of some components, which the authors correctly identify as avenues for future work. Overall, it is a substantial contribution that provides both a new technical tool and a valuable conceptual framework for the community.

---

# How to Evaluate Speech Translation with Source-Aware Neural MT Metrics

Authors: Mauro Cettolo, Marco Gaido, Matteo Negri, Sara Papi, Luisa Bentivogli

Keywords: Speech Translation Evaluation, Source-Aware Metrics, Synthetic Source Generation, Automatic Speech Recognition, Back-Translation, Cross-Lingual Re-Segmentation

Comments: None

Paper link: [http://arxiv.org/abs/2511.03295v1](http://arxiv.org/abs/2511.03295v1)

## Abstract

Automatic evaluation of speech-to-text translation (ST) systems is typically performed by comparing translation hypotheses with one or more reference translations. While effective to some extent, this approach inherits the limitation of reference-based evaluation that ignores valuable information from the source input. In machine translation (MT), recent progress has shown that neural metrics incorporating the source text achieve stronger correlation with human judgments. Extending this idea to ST, however, is not trivial because the source is audio rather than text, and reliable transcripts or alignments between source and references are often unavailable. In this work, we conduct the first systematic study of source-aware metrics for ST, with a particular focus on real-world operating conditions where source transcripts are not available. We explore two complementary strategies for generating textual proxies of the input audio, automatic speech recognition (ASR) transcripts, and back-translations of the reference translation, and introduce a novel two-step cross-lingual re-segmentation algorithm to address the alignment mismatch between synthetic sources and reference translations. Our experiments, carried out on two ST benchmarks covering 79 language pairs and six ST systems with diverse architectures and performance levels, show that ASR transcripts constitute a more reliable synthetic source than back-translations when word error rate is below 20%, while back-translations always represent a computationally cheaper but still effective alternative. Furthermore, our cross-lingual re-segmentation algorithm enables robust use of source-aware MT metrics in ST evaluation, paving the way toward more accurate and principled evaluation methodologies for speech translation.

## Summary

This paper addresses the challenge of evaluating speech translation (ST) systems using source-aware neural machine translation (MT) metrics like COMET and MetricX, which typically require source text that is unavailable in ST where the input is audio. The key contributions are a systematic investigation into using synthetic sources for ST evaluation and a novel cross-lingual re-segmentation algorithm (XLR-Segmenter) to align ASR transcripts with reference translations.

The authors explore two methods for generating synthetic textual sources: automatic speech recognition (ASR) transcripts and back-translations (BT) of the reference translations. They conduct extensive experiments on MuST-C and Europarl-ST datasets, covering 79 language pairs and six diverse ST systems. The methodology includes comparing synthetic sources under controlled conditions with known alignments, analyzing factors affecting their effectiveness, and evaluating the proposed XLR-Segmenter algorithm for real-world scenarios without audio-text alignments.

Key results show that both ASR and BT sources effectively substitute manual transcripts, with ASR generally preferable when WER ≤ 20%, while BT serves as a reliable alternative for higher WERs or when computational cost is a concern. The XLR-Segmenter algorithm, which refines segment boundaries using semantic embeddings from SimAlign, successfully aligns ASR outputs with reference translations, minimizing degradation in evaluation quality. The study demonstrates that source-aware metrics can be reliably applied to ST evaluation using synthetic sources, providing practical guidelines for their selection based on transcription quality, language coverage, and alignment requirements.

## Critique

Of course. Here is a detailed analysis of the strengths and weaknesses of the paper "How to Evaluate Speech Translation with Source-Aware Neural MT Metrics".

### Overall Summary

This is a highly rigorous and practically significant paper that addresses a critical, yet underexplored, methodological gap in Speech Translation (ST) evaluation. The work is distinguished by its systematic and exhaustive experimentation, leading to clear, actionable guidelines for the research community.

---

### Strengths

1.  **High Novelty and Clear Problem Definition:** The paper tackles a fundamental but non-trivial problem: how to leverage powerful, source-aware Machine Translation (MT) metrics (like COMET and MetricX) for ST when the "source" is audio, not text. It clearly defines three research questions (RQs) that structure the entire investigation, making the contribution easy to follow and assess.

2.  **Systematic and Exhaustive Experimental Design:** This is the paper's greatest strength. The authors leave no stone unturned in their validation:
    *   **Diverse Data:** Uses two large-scale, multilingual benchmarks (MuST-C and Europarl-ST) covering 79 language pairs, ensuring robustness across domains and languages.
    *   **Multiple Systems:** Evaluates six ST systems with different architectures (cascaded vs. direct) and performance levels, preventing conclusions from being biased by a single model's behavior.
    *   **Comprehensive Comparison:** Systematically compares two methods for generating synthetic sources (ASR transcripts vs. Back-Translation) using multiple state-of-the-art models for each (Whisper, OWSM, SeamlessM4T for ASR; MADLAD, NLLB for BT).
    *   **Controlled vs. "In-the-Wild":** The experimental progression is logical, starting from an idealized scenario (known audio-reference alignment) and progressively moving to a realistic, challenging setting (automatic segmentation and re-alignment).

3.  **Significant and Actionable Results:** The findings are not just statistically sound but are immediately useful for practitioners:
    *   **The 20% WER Threshold:** The identification of a clear, empirical threshold (WER ≤ 20%) for when ASR sources are preferable to BT sources is a major, practical contribution.
    *   **Effectiveness of XLR-Segmenter:** The proposed two-stage re-segmentation algorithm is shown to be highly effective, nearly closing the performance gap with manual segmentation. Releasing the code is a valuable contribution to the community.
    *   **Nuanced Insights:** The paper goes beyond a simple "which is better" answer. It provides nuanced insights, such as the observation that COMET is surprisingly robust to source degradation, while MetricX is more sensitive, and the important caveat about "biased conditions" when the ASR model is part of the evaluated system.

4.  **Clarity of Presentation:** The paper is exceptionally well-written and structured.
    *   The division of experiments into clear blocks with "Outline," "Results," and "Key observations" sections makes the dense experimental data digestible.
    *   The use of tables and figures is effective and supports the narrative.
    *   The discussion in Section 6 synthesizes the findings into a clear, decision-tree-like guide, discussing trade-offs between ASR and BT beyond just correlation (e.g., cost, language coverage, neutrality).

---

### Weaknesses

1.  **Limited Scope of "In-the-Wild" Conditions:** While the paper makes a strong effort to simulate real-world conditions, there are limitations:
    *   The audio data in both benchmarks is still relatively clean (TED Talks, parliamentary speeches). The findings may not fully hold for noisy, conversational, or multi-speaker audio, where ASR quality (WER) would likely be much worse, pushing the recommendation firmly towards BT.
    *   The paper acknowledges this limitation, but it remains a boundary for the generalizability of the results.

2.  **Focus on High-Resource Languages:** The study, by its own admission, is confined to high- and medium-resource languages. The performance of these methods on truly low-resource languages, where both ASR and MT quality can be poor and unstable, is an open question. The finding that BT is often more readily available for many languages is a crucial point, but its effectiveness in a low-resource context isn't tested here.

3.  **Computational Cost of the Full ASR Pipeline:** The paper quantitatively shows (in Appendix B) that the full ASR + XLR-Segmenter pipeline is significantly more computationally expensive than simple BT. While the correlation gains with a good ASR are clear, this practical barrier of higher cost could be a deciding factor for many researchers and is a valid weakness of the ASR approach.

4.  **Minor Point: MetricX Focus:** Given that MetricX was shown to be more sensitive to the source, the main body of the paper (Sections 5.2 and 5.4) focuses on it, relegating COMET results to an appendix. This is a valid choice for narrative clarity. However, it slightly under-represents the fact that for users of the very popular COMET metric, the choice of synthetic source matters less, which is an interesting finding in itself.

### Conclusion

This paper is a model of thorough, applied NLP research. Its primary strength lies in its comprehensive methodology, which allows it to derive strong, evidence-based conclusions. The proposed 20% WER rule-of-thumb and the effective XLR-Segmenter algorithm are significant contributions that will directly influence best practices in ST evaluation. The weaknesses are largely related to the boundaries of the study's scope (data cleanliness, language resources) rather than flaws in its execution, and the authors are transparent about these limitations. It is a highly valuable paper that successfully bridges a gap between MT and ST evaluation methodologies.

---

# From Insight to Exploit: Leveraging LLM Collaboration for Adaptive Adversarial Text Generation

Authors: Najrin Sultana, Md Rafi Ur Rashid, Kang Gu, Shagufta Mehnaz

Keywords: Adversarial Text Generation, LLM Collaboration, Dynamic Deceptor, Static Deceptor, Robustness Evaluation, Transferability, Defense Mechanisms

Comments: Findings of the Association for Computational Linguistics: EMNLP 2025
  (camera-ready)

Paper link: [http://arxiv.org/abs/2511.03128v1](http://arxiv.org/abs/2511.03128v1)

## Abstract

LLMs can provide substantial zero-shot performance on diverse tasks using a simple task prompt, eliminating the need for training or fine-tuning. However, when applying these models to sensitive tasks, it is crucial to thoroughly assess their robustness against adversarial inputs. In this work, we introduce Static Deceptor (StaDec) and Dynamic Deceptor (DyDec), two innovative attack frameworks designed to systematically generate dynamic and adaptive adversarial examples by leveraging the understanding of the LLMs. We produce subtle and natural-looking adversarial inputs that preserve semantic similarity to the original text while effectively deceiving the target LLM. By utilizing an automated, LLM-driven pipeline, we eliminate the dependence on external heuristics. Our attacks evolve with the advancements in LLMs and demonstrate strong transferability across models unknown to the attacker. Overall, this work provides a systematic approach for the self-assessment of an LLM's robustness. We release our code and data at https://github.com/Shukti042/AdversarialExample.

## Summary

This paper introduces two novel attack frameworks, **Dynamic Deceptor (DyDec)** and **Static Deceptor (StaDec)**, for generating adaptive adversarial examples against Large Language Models (LLMs) in sensitive classification tasks such as spam detection, hate speech detection, toxic comment classification, and fake news detection.

**Key Contributions:**
1. **Automated, LLM-driven attack pipelines** that eliminate reliance on external heuristics or manual adjustments. DyDec uses a multi-agent collaboration approach with specialized LLM roles (Reasoning, Red, Attacking, and Similarity Checker LLMs), while StaDec is a more lightweight version where the Attacking LLM handles all responsibilities.
2. **Generation of subtle, natural-looking adversarial examples** that preserve semantic similarity to original inputs while effectively deceiving target LLMs.
3. **Strong transferability** - adversarial examples successfully deceive models not involved in their creation.
4. **Evolutionary design** where attack effectiveness improves with advancements in LLM capabilities.

**Methods:**
- DyDec employs an iterative feedback process: Reasoning LLM analyzes target model predictions, Red LLM generates dynamic adversarial instructions, Attacking LLM crafts adversarial examples, and Similarity Checker LLM ensures semantic preservation.
- Attacks were evaluated on GPT-4o and Llama-3-70B across four datasets, with up to 8 feedback iterations per attack.

**Results:**
- Both DyDec and StaDec significantly outperformed existing attacks (PromptAttack and CombinedAttack), achieving attack success rates up to 100% on some tasks.
- In black-box settings where attackers had no knowledge of the target model, DyDec maintained consistent success rates (47.81-93.10%).
- The attacks demonstrated strong transferability, with success rates up to 100% against models not involved in adversarial generation.
- Existing defenses (perplexity-based detection, LLM-based detection, and paraphrasing) showed limited effectiveness against these attacks, highlighting the need for improved robustness measures.

The work provides a systematic approach for self-assessment of LLM robustness and exposes critical vulnerabilities in current LLM security frameworks.

## Critique

Of course. Here is a critique of the paper "From Insight to Exploit: Leveraging LLM Collaboration for Adaptive Adversarial Text Generation."

### Summary of Strengths

1.  **High Novelty and Conceptual Elegance:** The core idea—using a team of specialized LLMs to attack a target LLM—is highly innovative. Framing the problem as a multi-agent system (with Reasoning, Red, and Attacking LLMs) is a clever and intuitive way to automate the adversarial example generation process, moving beyond static, hand-crafted perturbations.
2.  **Strong and Comprehensive Empirical Results:** The paper demonstrates impressive results. The attack success rates for both Dynamic Deceptor (DyDec) and Static Deceptor (StaDec) significantly outperform existing benchmarks (PromptAttack and CombinedAttack) across multiple models (GPT-4o, Llama-3-70B) and diverse, sensitive tasks (spam, hate speech, toxicity, fake news). The high transferability of the generated examples is a particularly significant and concerning finding, highlighting a general vulnerability in LLMs.
3.  **Practical Threat Model and Black-Box Evaluation:** The threat model is realistic, assuming only query access to the target model, which aligns with how many real-world LLM APIs operate. The successful black-box evaluation, where the attacker uses a different model (GPT-4o) to attack the target (Llama-3-70B), strengthens the paper's claim about the attack's broad applicability.
4.  **Systematic Defense Evaluation:** The paper goes beyond just presenting a new attack by thoroughly evaluating it against three state-of-the-art defense methods (Perplexity-based, LLM-based, and Paraphrasing). Showing that these defenses are largely ineffective against DyDec/StaDec is a crucial contribution that underscores the severity of the vulnerability discovered.
5.  **Clear Presentation and Structure:** The paper is well-organized. The distinction between DyDec and StaDec is clear, and the use of figures and algorithm pseudocode effectively illustrates the complex, multi-step process.

### Summary of Weaknesses

1.  **Limited Discussion of the "Static" Baseline:** While StaDec is presented as a more lightweight alternative, its success—sometimes rivaling or even exceeding DyDec—raises important questions. The authors note that StaDec sometimes produces shorter, less faithful examples, but a deeper analysis is needed. Why is a simpler, non-reasoning-based approach so effective? This finding could suggest that the core vulnerability is the Attacking LLM's ability to generate persuasive text, with the reasoning component providing only a marginal, task-dependent benefit.
2.  **Ambiguity in "Semantic Similarity":** The paper relies on an LLM-as-a-judge for semantic similarity, which is a reasonable but potentially problematic choice. While Appendix C justifies this by showing failures of BERTScore and cosine similarity, it doesn't fully address the subjectivity and potential bias introduced by the Similarity Checker LLM. The high similarity scores for examples that seem to have been significantly altered or neutralized (e.g., in the hate speech task) warrant a more critical discussion. Human evaluation of similarity would have strengthened this claim.
3.  **Superficial Analysis of "Why" the Attacks Work:** The paper excellently demonstrates *that* the attacks work but provides less insight into *why*. A deeper analysis of what the Reasoning LLM identifies and how the Red LLM's instructions evolve would be invaluable. For instance, are the attacks exploiting specific linguistic patterns, logical fallacies, or biases in the target LLMs' decision boundaries? This mechanistic understanding is key for developing more robust defenses.
4.  **Cost and Scalability as a Significant Limitation:** The authors correctly identify cost as a major limitation. DyDec costing $0.07 per input on GPT-4o is prohibitively expensive for large-scale attacks or defense training. While they argue costs will decrease, this currently limits the practical immediacy of the threat and the feasibility of using this method for large-scale adversarial training.
5.  **Incomplete Defense Discussion:** While the evaluation of existing defenses is thorough, the proposed mitigation strategy in Section 7.4 is very high-level and speculative ("LLMs should learn to reason that generating messages... is effectively aiding spam"). It lacks concrete implementation ideas or experiments, leaving a gap between identifying the vulnerability and proposing a solution.

### Overall Assessment

This is a **strong and impactful paper**. Its primary strength lies in its novel, automated approach to generating adaptive adversarial examples, which proves to be highly effective and transferable against state-of-the-art LLMs. The results are significant because they expose a fundamental weakness in current LLM security that is not addressed by existing defenses.

The main weaknesses are related to the depth of analysis—specifically, a need to better understand the mechanisms behind the attack's success and a more critical examination of the "semantic similarity" metric. Despite these points, the paper makes a compelling case for a new class of LLM vulnerabilities and provides a powerful framework for future research in both offensive and defensive robustness.

