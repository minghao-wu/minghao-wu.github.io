---
title: "ArXiv Daily Digest on 2025-11-10"
collection: digests
permalink: /digests/arxiv_cs_CL_2025-11-10_report
date: 2025-11-10
location: "Online"
---

Today's research highlights two significant advances in efficient and multilingual language model development. The first paper introduces **Importance-Aware Data Selection**, proposing a novel **Model Instruction Weakness Value (MIWV)** metric that leverages **In-Context Learning (ICL)** to identify high-impact training samples, achieving superior performance with just 1% of data compared to full-dataset training. The second work, **"Beyond English,"** presents the **LMT (Large-scale Multilingual Translation)** model suite, addressing **English-centric bias** through innovative techniques like **Strategic Downsampling** to counter directional degeneration and **Parallel Multilingual Prompting (PMP)** for enhanced cross-lingual transfer, establishing new state-of-the-art results in multilingual machine translation with remarkable parameter efficiency.

## TL;DR

Total papers: 62 , Selected papers: 2

**TL;DR Summary of Recent arXiv Papers**

**Key Themes:** Data efficiency in LLM training, multilingual machine translation, and addressing English-centric bias in NLP systems.

**Paper Summaries:**

1. **Importance-Aware Data Selection for Efficient LLM Instruction Tuning** (https://arxiv.org/abs/2511.07074)
   - Proposes Model Instruction Weakness Value (MIWV) to identify high-quality instruction data
   - Uses in-context learning discrepancies to quantify data importance
   - Achieves SOTA performance with only 1% of data, outperforming full-dataset training
   - Universal method requiring no model training or external LLMs

2. **Beyond English: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs** (https://arxiv.org/abs/2511.07003)
   - Introduces LMT suite covering 60 languages with Chinese-English centric approach
   - Identifies "directional degeneration" problem in symmetric multi-way fine-tuning
   - Proposes Strategic Downsampling and Parallel Multilingual Prompting (PMP)
   - 4B model outperforms much larger models (NLLB-54B, Aya-101-13B)

**Main Insights:** Both papers focus on improving efficiency and quality in LLM applications through smarter data utilization - either by selecting the most impactful training data or by optimizing multilingual training strategies to overcome English-centric biases and directional degeneration issues.

---

# Importance-Aware Data Selection for Efficient LLM Instruction Tuning

Authors: Tingyu Jiang, Shen Li, Yiyao Song, Lan Zhang, Hualei Zhu, Yuan Zhao, Xiaohang Xu, Kenjiro Taura, Hao Henry Wang

Keywords: Data Selection, Instruction Tuning, LLM Efficiency, Model Instruction Weakness Value, In-Context Learning

Comments: Accepted by AAAI 2026 Oral

Paper link: [http://arxiv.org/abs/2511.07074v1](http://arxiv.org/abs/2511.07074v1)

## Abstract

Instruction tuning plays a critical role in enhancing the performance and efficiency of Large Language Models (LLMs). Its success depends not only on the quality of the instruction data but also on the inherent capabilities of the LLM itself. Some studies suggest that even a small amount of high-quality data can achieve instruction fine-tuning results that are on par with, or even exceed, those from using a full-scale dataset. However, rather than focusing solely on calculating data quality scores to evaluate instruction data, there is a growing need to select high-quality data that maximally enhances the performance of instruction tuning for a given LLM. In this paper, we propose the Model Instruction Weakness Value (MIWV) as a novel metric to quantify the importance of instruction data in enhancing model's capabilities. The MIWV metric is derived from the discrepancies in the model's responses when using In-Context Learning (ICL), helping identify the most beneficial data for enhancing instruction tuning performance. Our experimental results demonstrate that selecting only the top 1\% of data based on MIWV can outperform training on the full dataset. Furthermore, this approach extends beyond existing research that focuses on data quality scoring for data selection, offering strong empirical evidence supporting the effectiveness of our proposed method.

## Summary

This paper introduces **Importance-Aware Data Selection for Efficient LLM Instruction Tuning**, proposing a novel metric called **Model Instruction Weakness Value (MIWV)** to identify high-quality instruction data that maximizes performance gains during instruction tuning.

**Key Contributions:**
- A universal, fully automated data selection method that requires no model training or external LLMs
- The MIWV metric, which quantifies instruction sample importance by leveraging the model's inherent in-context learning capabilities
- Demonstrated effectiveness across multiple models and datasets with minimal data requirements

**Method:**
The approach involves three main steps:
1. **One-shot example retrieval**: For each instruction sample, find the most similar sample using cosine similarity of embeddings
2. **MIWV computation**: Calculate the loss difference between LLM responses with and without the one-shot example: `MIWV = Lθ(yi|xi,C) - Lθ(yi|xi)`
3. **High-quality data selection**: Rank samples by MIWV and select top-scoring ones for instruction tuning

A high MIWV indicates that the model performs poorly on that instruction type even with contextual help, making these samples valuable for improving model capabilities.

**Results:**
The method achieves remarkable efficiency:
- Using only **1%** of data from Alpaca and WizardLM datasets outperforms full-dataset training
- Models trained with MIWV-selected data show superior performance on benchmarks including Open LLM Leaderboard and AlpacaEval
- Outperforms eight existing data selection methods in win rate comparisons
- Maintains effectiveness across different model architectures (LLaMA, Qwen2.5) and embedding models

The approach provides a cost-effective solution for instruction tuning, demonstrating that carefully selected small datasets can surpass the performance of full-scale training while significantly reducing computational resources.

## Critique

This paper presents "Importance-Aware Data Selection for Efficient LLM Instruction Tuning" and introduces the Model Instruction Weakness Value (MIWV) metric for data selection. Here's my assessment:

**Strengths:**

**Novelty and Approach:**
- The MIWV metric is genuinely innovative, leveraging in-context learning (ICL) to quantify instruction sample importance through loss differences between prompted and unprompted responses
- The approach is model-agnostic and requires no external LLMs or training, making it computationally efficient
- The combination of one-shot example retrieval with loss-based importance scoring provides a unique perspective on data selection

**Significant Results:**
- The key finding that training with only 1% of data can outperform full-dataset training is compelling and has substantial practical implications
- Extensive experiments across multiple datasets (Alpaca, WizardLM, NIV2) and model families (LLaMA, Qwen) demonstrate strong generalization
- Superior performance compared to multiple state-of-the-art methods (IFD Score, SelectIT, Deita, etc.) in both effectiveness and efficiency

**Methodological Rigor:**
- Comprehensive ablation studies validate the importance of each component (MIWV vs. prompt loss alone, embedding model variations)
- Cross-model validation on different architectures (LLaMA vs. Qwen) strengthens claims of universality
- Multi-dimensional analysis of selected data characteristics provides good interpretability

**Weaknesses:**

**Technical Concerns:**
- The claim that MIWV identifies "model weaknesses" is somewhat speculative - the metric might simply identify samples where ICL introduces confusion rather than true capability gaps
- Limited discussion of potential failure cases or scenarios where the approach might not work well
- The efficiency advantage (85 minutes) is significant but not dramatically better than some baselines like Superfiltering (8 minutes)

**Presentation Issues:**
- The mathematical formulation could be clearer, particularly the relationship between MIWV values and actual model improvement
- Some figures and tables are referenced but not included in the provided text, making evaluation difficult
- The paper could benefit from more detailed analysis of why certain data ratios (1%, 5%, etc.) perform best in different scenarios

**Conceptual Limitations:**
- The approach assumes that samples causing ICL confusion are beneficial for instruction tuning, but this relationship isn't thoroughly justified theoretically
- Limited discussion of how this method scales to extremely large datasets or different types of instruction formats

**Overall Assessment:**
This is a strong paper with a novel, practical approach to data selection that demonstrates impressive empirical results. The MIWV metric represents a meaningful contribution to the field of efficient LLM training. While some theoretical foundations could be strengthened, the extensive experimental validation and significant performance improvements make this a valuable contribution to instruction tuning research. The method's simplicity and model-agnostic nature enhance its potential for broad adoption.

---

# Beyond English: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs

Authors: Yingfeng Luo, Ziqiang Xu, Yuxuan Ouyang, Murun Yang, Dingyang Lin, Kaiyan Chang, Tong Zheng, Bei Li, Peinan Feng, Quan Du, Tong Xiao, Jingbo Zhu

Keywords: Multilingual Machine Translation, Large Language Models, Directional Degeneration, Strategic Downsampling, Parallel Multilingual Prompting, Chinese-English-Centric, Cross-Lingual Transfer

Comments: None

Paper link: [http://arxiv.org/abs/2511.07003v1](http://arxiv.org/abs/2511.07003v1)

## Abstract

Large language models have significantly advanced Multilingual Machine Translation (MMT), yet the broad language coverage, consistent translation quality, and English-centric bias remain open challenges. To address these challenges, we introduce \textbf{LMT}, a suite of \textbf{L}arge-scale \textbf{M}ultilingual \textbf{T}ranslation models centered on both Chinese and English, covering 60 languages and 234 translation directions. During development, we identify a previously overlooked phenomenon of \textbf{directional degeneration}, where symmetric multi-way fine-tuning data overemphasize reverse directions (X $\to$ En/Zh), leading to excessive many-to-one mappings and degraded translation quality. We propose \textbf{Strategic Downsampling}, a simple yet effective method to mitigate this degeneration. In addition, we design \textbf{Parallel Multilingual Prompting (PMP)}, which leverages typologically related auxiliary languages to enhance cross-lingual transfer. Through rigorous data curation and refined adaptation strategies, LMT achieves SOTA performance among models of comparable language coverage, with our 4B model (LMT-60-4B) surpassing the much larger Aya-101-13B and NLLB-54B models by a substantial margin. We release LMT in four sizes (0.6B/1.7B/4B/8B) to catalyze future research and provide strong baselines for inclusive, scalable, and high-quality MMT \footnote{\href{https://github.com/NiuTrans/LMT}{https://github.com/NiuTrans/LMT}}.

## Summary

Based on the provided paper "Beyond English: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs," here is a summary focusing on its key contributions, methods, and results:

This paper introduces LMT, a suite of Large-scale Multilingual machine Translation models designed to address the prevalent English-centric bias in current systems by centering on both Chinese and English. LMT covers 60 languages and 234 translation directions, supporting English ↔ 59 languages and Chinese ↔ 58 languages. The development follows a standard Continued Pre-training (CPT) → Supervised Fine-tuning (SFT) pipeline, built upon the Qwen3 backbone.

The key contributions are threefold. First, the authors identify and analyze a previously overlooked issue termed **"directional degeneration,"** where symmetric multi-way fine-tuning data overemphasize reverse directions (X → En/Zh), leading to excessive many-to-one mappings and degraded translation quality. To mitigate this, they propose **Strategic Downsampling**, a simple yet effective method that retains only a small percentage (e.g., 5%) of the reverse direction data during SFT. Second, they introduce **Parallel Multilingual Prompting (PMP)**, a technique that augments the translation instruction with a parallel sentence from a typologically related auxiliary language (or English for Chinese-centric cases) to enhance cross-lingual transfer, particularly for low-resource languages. Third, the paper releases the **LMT model suite** in four sizes (0.6B/1.7B/4B/8B) to serve as a strong baseline for inclusive and scalable MMT.

The methodology involves rigorous data curation, including large-scale collection from sources like OPUS, pseudo-parallel synthesis, and multi-dimensional filtering to create a high-quality corpus. For adaptation, CPT is performed on a balanced mixture of monolingual and bilingual data, followed by SFT that incorporates the proposed Strategic Downsampling and PMP techniques.

The results demonstrate that LMT achieves state-of-the-art performance among models of comparable language coverage. Notably, the LMT-60-4B model surpasses much larger models like Aya-101-13B and NLLB-54B by a substantial margin, showing exceptional parameter efficiency. Ablation studies confirm the individual contributions of Strategic Downsampling, CPT, and PMP, with Strategic Downsampling alone providing remarkable improvements (e.g., +11.45 COMET points in X→Zh direction). Analyses also show that PMP enhances zero-shot transfer capabilities and that self-generated auxiliary hints (PMP-S) can be effectively used at inference time. The work concludes that LMT provides a robust, high-quality baseline for large-scale, inclusive multilingual machine translation.

## Critique

Of course. Here is a critique of the paper "Beyond English: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs," focusing on its strengths, weaknesses, and overall contribution.

### Summary of Strengths

1.  **Addressing a Critical and Underexplored Problem:** The paper's core mission—to move beyond English-centric multilingual machine translation (MMT) by building a Chinese-English-centric model—is highly significant. It tackles a real-world limitation in current LLM-based MT systems and addresses the data scarcity for non-English-centric pairs, particularly Chinese.

2.  **Identification and Analysis of "Directional Degeneration":** This is arguably the most novel contribution. The paper not only identifies a clear performance pathology (degradation in X→En/Zh directions) in standard multi-way SFT but also provides a compelling hypothesis (the "Shallow Mapping Trap" due to excessive many-to-one mappings) and a simple, effective, and data-efficient solution (Strategic Downsampling). The analysis is rigorous, with controlled experiments showing the phenomenon's generality across base models and its scaling with the number of languages.

3.  **Practical and Effective Methodologies:** Both proposed methods are elegant and practical:
    *   **Strategic Downsampling:** It is remarkably simple, requiring only a minimal retention rate (5%) of reverse-direction data to prevent collapse, making it easy to adopt.
    *   **Parallel Multilingual Prompting (PMP):** This is a clever way to explicitly encourage cross-lingual transfer. Its design, using typologically similar languages for En↔X and English for Zh↔X, is well-motivated. The inference-time analysis showing that self-generated prompts can outperform oracle prompts is a fascinating and useful finding.

4.  **Comprehensive System Building and Evaluation:** The work goes beyond a narrow technical contribution by building and releasing a full model suite, **LMT**, in four sizes covering 60 languages and 234 directions. The evaluation is extensive, comparing against a wide array of strong baselines (both general-purpose and dedicated MMT models) and demonstrating state-of-the-art or highly competitive results, especially considering the model's parameter efficiency (e.g., LMT-4B outperforming NLLB-54B).

5.  **Clarity and Thoroughness:** The paper is generally well-written and structured. The figures effectively illustrate key concepts like the directional degeneration phenomenon and the ablation study results. The inclusion of extensive appendices (hyperparameters, data sources, full results) adds to the work's credibility and reproducibility.

### Summary of Weaknesses

1.  **Limited Theoretical Justification for PMP's Success:** While PMP is shown to work empirically, the paper provides less theoretical insight into *why* it works so well. The mechanism of how the auxiliary sentence guides the model "toward higher-fidelity translations" is described intuitively but could be probed deeper, for instance, by analyzing attention patterns or representation spaces during PMP inference.

2.  **Ablation Study Could Be More Granular:** The ablation study in Figure 5 is excellent for showing the cumulative impact of each component. However, it would be even more informative to see the individual effect of PMP in isolation (e.g., Base+CPT+SFT+SD vs. Base+CPT+SFT+SD+PMP) to disentangle its contribution from the foundational CPT and SD steps more clearly.

3.  **Potential Overlap with Prior Work:** The paper correctly cites a concurrent work (Zheng et al., 2025) that also observed asymmetric degradation. While this work goes significantly further by identifying the root cause and providing a data-level (not model-level) solution, the framing could more explicitly delineate the key advancements over this prior observation to avoid any perception of incrementalism.

4.  **Evaluation on a Single Benchmark:** The primary evaluation relies on FLORES-200, a high-quality but somewhat narrow academic benchmark. While the authors mention WMT24++ results in the appendix, a broader evaluation on real-world, noisy, or domain-specific text would strengthen the claims about the model's robustness and practical utility.

### Overall Assessment

This is a **strong and impactful paper**. It makes a significant contribution to the field of multilingual machine translation by systematically addressing a major bottleneck (English-centric bias) and a subtle but critical training pitfall (directional degeneration). The proposed solutions are not only novel and insightful but also simple and practical, making them highly adoptable.

The significance of the results is high, as demonstrated by the performance of the released LMT models, which provide a powerful new baseline for inclusive MMT. The presentation is clear and thorough, effectively communicating both the technical details and the broader implications of the work. The weaknesses are relatively minor and point to opportunities for future research rather than fundamental flaws.

