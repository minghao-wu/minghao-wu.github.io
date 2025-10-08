---
title: "Learning to Summarize by Learning to Quiz: Adversarial Agentic Collaboration for Long Document Summarization"
collection: publications
permalink: /publication/arxiv-2025-summq
date: 2025-09-26
venue: 'CoRR'
paperurl: 'https://arxiv.org/abs/2509.20900'
paperurltext: 'Link to arXiv'
citation: 'Weixuan Wang*, <b>Minghao Wu*</b>, Barry Haddow, and Alexandra Birch. <a href="https://arxiv.org/abs/2509.20900"><u>Learning to Summarize by Learning to Quiz: Adversarial Agentic Collaboration for Long Document Summarization</u></a>. abs/2509.20900.'
---

```
@article{wang-etal-2025-hbo,
  author       = {Weixuan Wang and Minghao Wu and Barry Haddow and Alexandra Birch},
  title        = {Learning to Summarize by Learning to Quiz: Adversarial Agentic Collaboration for Long Document Summarization},
  journal      = {CoRR},
  volume       = {abs/2509.20900},
  year         = {2025},
  url          = {https://arxiv.org/abs/2509.20900},
  eprinttype   = {arXiv},
  eprint       = {2509.20900}
}
```

## Abstract
Long document summarization remains a significant challenge for current large language models (LLMs), as existing approaches commonly struggle with information loss, factual inconsistencies, and coherence issues when processing excessively long documents. We propose SummQ, a novel adversarial multi-agent framework that addresses these limitations through collaborative intelligence between specialized agents operating in two complementary domains: summarization and quizzing. Our approach employs summary generators and reviewers that work collaboratively to create and evaluate comprehensive summaries, while quiz generators and reviewers create comprehension questions that serve as continuous quality checks for the summarization process. This adversarial dynamic, enhanced by an examinee agent that validates whether the generated summary contains the information needed to answer the quiz questions, enables iterative refinement through multifaceted feedback mechanisms. We evaluate SummQ on three widely used long document summarization benchmarks. Experimental results demonstrate that our framework significantly outperforms existing state-of-the-art methods across ROUGE and BERTScore metrics, as well as in LLM-as-a-Judge and human evaluations. Our comprehensive analyses reveal the effectiveness of the multi-agent collaboration dynamics, the influence of different agent configurations, and the impact of the quizzing mechanism. This work establishes a new approach for long document summarization that uses adversarial agentic collaboration to improve summarization quality.