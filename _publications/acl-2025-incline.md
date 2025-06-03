---
title: "Bridging the Language Gaps in Large Language Models with Inference-Time Cross-Lingual Intervention"
collection: publications
permalink: /publication/acl-2025-incline
date: 2025-05-15
venue: 'Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)'
paperurl: 'https://arxiv.org/abs/2410.12462'
paperurltext: 'Link to arXiv'
citation: 'Weixuan Wang*, <b>Minghao Wu*</b>, Barry Haddow, and Alexandra Birch. <b>ACL 2025</b>. [Link to arXiv](https://arxiv.org/abs/2410.12462)'
---

```
@inproceedings{wang-etal-2024-bridge,
  author       = {Weixuan Wang and Minghao Wu and Barry Haddow and Alexandra Birch},
  title        = {Bridging the Language Gaps in Large Language Models with Inference-Time Cross-Lingual Intervention},
  journal      = {CoRR},
  volume       = {abs/2410.12462},
  year         = {2024},
  url          = {https://arxiv.org/abs/2410.12462},
  eprinttype   = {arXiv},
  eprint       = {2410.12462}
}
```

## Abstract
Large Language Models (LLMs) have shown remarkable capabilities in natural language processing but exhibit significant performance gaps among different languages. Most existing approaches to address these disparities rely on pretraining or fine-tuning, which are resource-intensive. To overcome these limitations without incurring significant costs, we propose Inference-Time Cross-Lingual Intervention (INCLINE), a novel framework that enhances LLM performance on low-performing (source) languages by aligning their internal representations with those of high-performing (target) languages during inference. INCLINE initially learns alignment matrices using parallel sentences from source and target languages through a Least-Squares optimization, and then applies these matrices during inference to transform the low-performing language representations toward the high-performing language space. Extensive experiments on nine benchmarks with five LLMs demonstrate that INCLINE significantly improves performance across diverse tasks and languages, compared to recent strong baselines. Our analysis demonstrates that INCLINE is highly cost-effective and applicable to a wide range of applications. In addition, we release the code to foster research along this line