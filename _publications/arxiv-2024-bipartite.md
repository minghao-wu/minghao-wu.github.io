---
title: "The Best of Both Worlds: Bridging Quality and Diversity in Data Selection with Bipartite Graph"
collection: publications
permalink: /publication/arxiv-2024-bipartite
date: 2024-10-16
venue: 'CoRR'
paperurl: 'https://arxiv.org/abs/2410.12458'
paperurltext: 'Link to arXiv'
citation: '<b>Minghao Wu</b>, Thuy-Trang Vu, Lizhen Qu, Gholamreza Haffari. 2024. <a href="http://minghao-wu.github.io/files/papers/bipartite_arxiv_2024.pdf"><u>The Best of Both Worlds: Bridging Quality and Diversity in Data Selection with Bipartite Graph</u></a>. In <i>CoRR</i>, abs/2410.12458.'
---

```
@article{wu2024best,
  author       = {Minghao Wu, 
                  Thuy-Trang Vu, 
                  Lizhen Qu, 
                  Gholamreza Haffari
                  },
  title        = {The Best of Both Worlds: Bridging Quality and Diversity in Data Selection with Bipartite Graph},
  journal      = {CoRR},
  volume       = {abs/2410.12458},
  year         = {2024},
  url          = {https://arxiv.org/abs/2410.12458},
  eprinttype   = {arXiv},
  eprint       = {2410.12458}
}
```

## Abstract
The performance of large language models (LLMs) in natural language processing (NLP) tasks is significantly influenced by the quality and diversity of data used for supervised fine-tuning (SFT). Current data selection methods often focus solely on quality or diversity, leading to underperforming models due to suboptimal training data. In this paper, we introduce GraphFilter, a novel method that represents the dataset as a bipartite graph, linking sentences to their constituent n-grams. This representation effectively captures the relationships between sentences and linguistic patterns, facilitating the selection of sentences that enhance n-gram diversity. To balance quality and diversity during selection, we propose a priority function that combines the quality metric with the diversity metric in a multiplicative manner. GraphFilter iteratively selects high-priority sentences, updates the bipartite graph by removing covered n-grams, and re-calculates priorities to reflect the evolving data landscape. We conduct extensive experiments using three model backbones across six widely used benchmarks. The results demonstrate that GraphFilter outperforms all nine baseline approaches, achieving superior model performance and computational efficiency. Our analyses validate the effectiveness of our design choices, examine the subsets selected by GraphFilter and other methods, highlight the importance of instruction diversity, and explore the role of quality and diversity in relation to subset sizes. GraphFilter establishes a new foundation for effective data selection strategies, encouraging further research in data selection for LLMs.