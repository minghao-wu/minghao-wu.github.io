---
title: "Adapting Large Language Models for Document-Level Machine Translation"
collection: publications
permalink: /publication/arxiv-2024-docnmt
date: 2024-01-12
venue: 'CoRR'
paperurl: 'https://arxiv.org/abs/2401.06468'
paperurltext: 'Link to arXiv'
citation: '<b>Minghao Wu</b>, Thuy-Trang Vu, Lizhen Qu, George Foster, Gholamreza Haffari. 2024. <a href="http://minghao-wu.github.io/files/papers/docnmt_arxiv_2024.pdf"><u>Adapting Large Language Models for Document-Level Machine Translation</u></a>. In <i>CoRR</i>, abs/2401.06468.'
---

```
@article{wu2024docnmt,
  author       = {Minghao Wu, 
                  Thuy-Trang Vu, 
                  Lizhen Qu, 
                  George Foster, 
                  Gholamreza Haffari
                  },
  title        = {Adapting Large Language Models for Document-Level Machine Translation},
  journal      = {CoRR},
  volume       = {abs/2401.06468},
  year         = {2024},
  url          = {https://arxiv.org/abs/2401.06468},
  eprinttype   = {arXiv},
  eprint       = {2401.06468}
}
```

## Abstract
Large language models (LLMs) have significantly advanced various natural language processing (NLP) tasks. Recent research indicates that moderately-sized LLMs often outperform larger ones after task-specific fine-tuning. This study focuses on adapting LLMs for document-level machine translation (DocMT) for specific language pairs. We first investigate the impact of prompt strategies on translation performance and then conduct extensive experiments using two fine-tuning methods, three LLM backbones, and 18 translation tasks across nine language pairs. Our results show that specialized models can sometimes surpass GPT-4 in translation performance but still face issues like off-target translation due to error propagation in decoding. We provide an in-depth analysis of these LLMs tailored for DocMT, examining translation errors, discourse phenomena, training strategies, the scaling law of parallel documents, recent test set evaluations, and zero-shot crosslingual transfer. Our findings highlight the strengths and limitations of LLM-based DocMT models and provide a foundation for future research.