---
title: "(Perhaps) Beyond Human Translation: Harnessing Multi-Agent Collaboration for Translating Ultra-Long Literary Texts"
collection: publications
permalink: /publication/tacl-2025-transagents
date: 2025-7-27
venue: 'Transactions of the Association for Computational Linguistics'
paperurl: 'https://arxiv.org/abs/2405.11804'
paperurltext: 'Link to arXiv'
citation: '<b>Minghao Wu</b>, Jiahao Xu, Yulin Yuan, Gholamreza Haffari, Longyue Wang, Weihua Luo, and Kaifu Zhang. <b>TACL 2025</b>
---

```
@article{wu2024transagents,
  author       = {Minghao Wu, 
                  Jiahao Xu, 
                  Yulin Yuan, 
                  Gholamreza Haffari, 
                  Longyue Wang,
                  Weihua Luo, 
                  Kaifu Zhang
                  },
  title        = {(Perhaps) Beyond Human Translation: Harnessing Multi-Agent Collaboration for Translating Ultra-Long Literary Texts},
  journal      = {Transactions of the Association for Computational Linguistics},
  volume       = {abs/2405.11804},
  year         = {2024},
  url          = {https://arxiv.org/abs/2405.11804},
  eprinttype   = {arXiv},
  eprint       = {2405.11804}
}
```

## Abstract
Literary translation remains one of the most challenging frontiers in machine translation due to the complexity of capturing figurative language, cultural nuances, and unique stylistic elements. In this work, we introduce TransAgents, a novel multi-agent framework that simulates the roles and collaborative practices of a human translation company, including a CEO, Senior Editor, Junior Editor, Translator, Localization Specialist, and Proofreader. The translation process is divided into two stages: a preparation stage where the team is assembled and comprehensive translation guidelines are drafted, and an execution stage that involves sequential translation, localization, proofreading, and a final quality check. Furthermore, we propose two innovative evaluation strategies: Monolingual Human Preference (MHP), which evaluates translations based solely on target language quality and cultural appropriateness, and Bilingual LLM Preference (BLP), which leverages large language models like GPT-4} for direct text comparison. Although TransAgents achieves lower d-BLEU scores, due to the limited diversity of references, its translations are significantly better than those of other baselines and are preferred by both human evaluators and LLMs over traditional human references and GPT-4} translations. Our findings highlight the potential of multi-agent collaboration in enhancing translation quality, particularly for longer texts.