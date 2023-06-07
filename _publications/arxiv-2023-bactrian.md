---
title: "Bactrian-X : A Multilingual Replicable Instruction-Following Model with Low-Rank Adaptation"
collection: publications
permalink: /publication/arxiv-2023-bactrian
date: 2023-05-24
venue: 'CoRR'
paperurl: 'https://arxiv.org/abs/2305.15011'
paperurltext: 'Link to arXiv'
citation: 'Haonan Li, Fajri Koto, <b>Minghao Wu</b>, Alham Fikri Aji, Timothy Baldwin. 2023. <a href="http://minghao-wu.github.io/files/papers/bactrian_arxiv_2023.pdf"><u>Bactrian-X : A Multilingual Replicable Instruction-Following Model with Low-Rank Adaptation</u></a>. In <i>CoRR</i>, abs/2305.15011.'
---

```
@article{lamini-lm,
  author       = {Haonan Li and 
                  Fajri Koto and 
                  Minghao Wu and 
                  Alham Fikri Aji and 
                  Timothy Baldwin
                  },
  title        = {Bactrian-X : A Multilingual Replicable Instruction-Following Model with Low-Rank Adaptation},
  journal      = {CoRR},
  volume       = {abs/2305.15011},
  year         = {2023},
  url          = {https://arxiv.org/abs/2305.15011},
  eprinttype   = {arXiv},
  eprint       = {2305.15011}
}
```

## Abstract
Instruction tuning has shown great promise in the field of natural language processing. However, the research on multilingual instruction tuning has been limited due to the scarcity of high-quality instruction-response datasets. To address this gap, we present Bactrian-X, a comprehensive multilingual parallel dataset of 3.4 million instruction-response pairs across 52 languages. Leveraging this dataset, we train a set of adapters using low-rank adaptation (LoRA), which are lightweight components seamlessly integrated with foundational models. These adapters have a significantly smaller parameter count than the base model, making them easily replaceable and usable as plug-ins for different languages or language groups. Through extensive experiments on 52 languages, we demonstrate the superior performance of our models in various multilingual evaluation settings. Our proposed models outperform both the vanilla models and the existing instruction-tuned models.