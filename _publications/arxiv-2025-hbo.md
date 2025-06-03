---
title: "HBO: Hierarchical Balancing Optimization for Fine-Tuning Large Language Models"
collection: publications
permalink: /publication/arxiv-2025-hbo
date: 2025-05-26
venue: 'CoRR'
paperurl: 'https://arxiv.org/abs/2505.12300'
paperurltext: 'Link to arXiv'
citation: 'Weixuan Wang*, <b>Minghao Wu*</b>, Barry Haddow, and Alexandra Birch. 2025. <a href="http://minghao-wu.github.io/files/papers/hbo_arxiv_2025.pdf"><u>HBO: Hierarchical Balancing Optimization for Fine-Tuning Large Language Models</u></a>. abs/2505.12300.'
---

```
@article{wang-etal-2025-hbo,
  author       = {Weixuan Wang and Minghao Wu and Barry Haddow and Alexandra Birch},
  title        = {HBO: Hierarchical Balancing Optimization for Fine-Tuning Large Language Models},
  journal      = {CoRR},
  volume       = {abs/2505.12300},
  year         = {2025},
  url          = {https://arxiv.org/abs/2505.12300},
  eprinttype   = {arXiv},
  eprint       = {2505.12300}
}
```

## Abstract
Fine-tuning large language models (LLMs) on a mixture of diverse datasets poses challenges due to data imbalance and heterogeneity. Existing methods often address these issues across datasets (globally) but overlook the imbalance and heterogeneity within individual datasets (locally), which limits their effectiveness. We introduce Hierarchical Balancing Optimization (HBO), a novel method that enables LLMs to autonomously adjust data allocation during fine-tuning both across datasets (globally) and within each individual dataset (locally). HBO employs a bilevel optimization strategy with two types of actors: a Global Actor, which balances data sampling across different subsets of the training mixture, and several Local Actors, which optimizes data usage within each subset based on difficulty levels. These actors are guided by reward functions derived from the LLM's training state, which measure learning progress and relative performance improvement. We evaluate HBO on three LLM backbones across nine diverse tasks in multilingual and multitask setups. Results show that HBO consistently outperforms existing baselines, achieving significant accuracy gains. Our in-depth analysis further demonstrates that both the global actor and local actors of HBO effectively adjust data usage during fine-tuning. HBO provides a comprehensive solution to the challenges of data imbalance and heterogeneity in LLM fine-tuning, enabling more effective training across diverse datasets.