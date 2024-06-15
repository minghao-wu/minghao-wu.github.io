---
title: "Mixture-of-Skills: Learning to Optimize Data Usage for Fine-Tuning Large Language Models"
collection: publications
permalink: /publication/arxiv-2024-mos
date: 2024-06-13
venue: 'CoRR'
paperurl: 'https://arxiv.org/abs/2406.08811'
paperurltext: 'Link to arXiv'
citation: '<b>Minghao Wu</b>, Thuy-Trang Vu, Lizhen Qu, Gholamreza Haffari. 2024. <a href="http://minghao-wu.github.io/files/papers/mos_arxiv_2024.pdf"><u>Mixture-of-Skills: Learning to Optimize Data Usage for Fine-Tuning Large Language Models</u></a>. In <i>CoRR</i>, abs/2406.08811.'
---

```
@article{wu2024docnmt,
  author       = {Minghao Wu, 
                  Thuy-Trang Vu, 
                  Lizhen Qu, 
                  Gholamreza Haffari
                  },
  title        = {Mixture-of-Skills: Learning to Optimize Data Usage for Fine-Tuning Large Language Models},
  journal      = {CoRR},
  volume       = {abs/2406.08811},
  year         = {2024},
  url          = {https://arxiv.org/abs/2406.08811},
  eprinttype   = {arXiv},
  eprint       = {2406.08811}
}
```

## Abstract
Large language models (LLMs) are typically fine-tuned on diverse and extensive datasets sourced from various origins to develop a comprehensive range of skills, such as writing, reasoning, chatting, coding, and more. Each skill has unique characteristics, and these datasets are often heterogeneous and imbalanced, making the fine-tuning process highly challenging. Balancing the development of each skill while ensuring the model maintains its overall performance requires sophisticated techniques and careful dataset curation. In this work, we propose a general, model-agnostic, reinforcement learning framework, Mixture-of-Skills (MoS), that learns to optimize data usage automatically during the fine-tuning process. This framework ensures the optimal comprehensive skill development of LLMs by dynamically adjusting the focus on different datasets based on their current learning state. To validate the effectiveness of MoS, we conduct extensive experiments using three diverse LLM backbones on two widely used benchmarks and demonstrate that MoS substantially enhances model performance. Building on the success of MoS, we propose MoSpec, an adaptation for task-specific fine-tuning, which harnesses the utilities of various datasets for a specific purpose. Our work underlines the significance of dataset rebalancing and present MoS as a powerful, general solution for optimizing data usage in the fine-tuning of LLMs for various purposes.