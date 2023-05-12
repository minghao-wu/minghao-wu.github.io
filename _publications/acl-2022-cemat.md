---
title: "LaMini-LM: A Diverse Herd of Distilled Models from Large-Scale Instructions"
collection: publications
permalink: /publication/acl-2022-cemat
date: 2022-05-22
venue: 'Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)'
paperurl: 'https://aclanthology.org/2022.acl-long.442'
paperurltext: 'Link to ACL anthology'
citation: '<b>Minghao Wu</b>, Abdul Waheed, Chiyu Zhang, Muhammad Abdul-Mageed, Alham Fikri Aji. 2022. <a href="http://minghao-wu.github.io/files/papers/lamini_arxiv_2023.pdf"><u>LaMini-LM: A Diverse Herd of Distilled Models from Large-Scale Instructions</u></a>. In <i>Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)</i>, pages 6379–6391, Dublin, Ireland. Association for Computational Linguistics.'
---

```
@article{lamini-lm,
  author       = {Minghao Wu and
                  Abdul Waheed and
                  Chiyu Zhang and
                  Muhammad Abdul-Mageed and
                  Alham Fikri Aji
                  },
  title        = {LaMini-LM: A Diverse Herd of Distilled Models from Large-Scale Instructions},
  journal      = {CoRR},
  volume       = {abs/2304.14402},
  year         = {2023},
  url          = {https://arxiv.org/abs/2304.14402},
  eprinttype   = {arXiv},
  eprint       = {2304.14402}
}
```

## Abstract
Large language models (LLMs) with instruction finetuning demonstrate superior generative capabilities. However, these models are resource intensive. To alleviate this issue, we explore distilling knowledge from instruction-tuned LLMs to much smaller ones. To this end, we carefully develop a large set of 2.58M instructions based on both existing and newly-generated instructions. In addition to being sizeable, we design our instructions to cover a broad set of topics to ensure. A thorough investigation of our instruction data demonstrate their diversity, and we generate responses for these instructions using gpt-3.5-turbo. We then exploit the instructions to tune a host of models, dubbed LaMini-LM, of varying sizes, both from the encoder-decoder as well as the decoder-only families. We evaluate our models both automatically (on 15 different NLP benchmarks) and manually. Results show that our proposed LaMini-LM are on par with competitive baselines while being nearly 10 times smaller in size.