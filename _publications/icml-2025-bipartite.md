---
title: "The Best of Both Worlds: Bridging Quality and Diversity in Data Selection with Bipartite Graph"
collection: publications
permalink: /publication/icml-2025-bipartite
date: 2025-05-01
venue: 'Proceedings of the 42nd International Conference on Machine Learning'
paperurl: 'https://arxiv.org/abs/2410.12458'
paperurltext: 'Link to arXiv'
citation: '<b>Minghao Wu</b>, Thuy-Trang Vu, Lizhen Qu, and Gholamreza Haffari. <b>ICML 2025</b>. [Link to arXiv](https://arxiv.org/abs/2410.12458)'
---

```
@inproceedings{wu-etal-2024-perhaps,
  author       = {Minghao Wu, Thuy-Trang Vu, Lizhen Qu, Gholamreza Haffari},
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
The performance of large language models (LLMs) is strongly influenced by the quality and diversity of data used during supervised fine-tuning (SFT). However, current data selection methods often prioritize one aspect over the other, resulting in suboptimal training outcomes. To address this, we formulate data selection as a set cover problem and present GraphFilter, a novel approach that balances both quality and diversity in data selection. GraphFilter models the dataset as a bipartite graph connecting sentences to their constituent n-grams, then employs a priority function that combines quality and diversity metrics multiplicatively. GraphFilter iteratively selects sentences with the highest priority, removes covered n-grams from the bipartite graph, and recomputes priorities to reflect the changing data landscape. We validate GraphFilter using three model backbones across six widely-used benchmarks, demonstrating that it outperforms nine existing baselines in both model performance and computational efficiency. Further analysis shows that our design choices lead to more effective subset selection, underscores the value of instruction diversity, and provides insights into how quality and diversity interact with different subset sizes.