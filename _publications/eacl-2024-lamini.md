---
title: "LaMini-LM: A Diverse Herd of Distilled Models from Large-Scale Instructions"
collection: publications
permalink: /publication/eacl-2024-lamini
date: 2023-04-27
venue: 'Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)'
paperurl: 'https://aclanthology.org/2024.eacl-long.57/'
paperurltext: 'Link to ACL anthology'
citation: '<b>Minghao Wu</b>, Abdul Waheed, Chiyu Zhang, Muhammad Abdul-Mageed, and Alham Fikri Aji. 2024. <a href="http://minghao-wu.github.io/files/papers/lamini_eacl_2024.pdf"><u>LaMini-LM: A Diverse Herd of Distilled Models from Large-Scale Instructions</u></a>. In <i>Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)</i>, pages 944–964, St. Julian’s, Malta. Association for Computational Linguistics.'
---

```
@inproceedings{wu-etal-2024-lamini,
    title = "{L}a{M}ini-{LM}: A Diverse Herd of Distilled Models from Large-Scale Instructions",
    author = "Wu, Minghao  and
      Waheed, Abdul  and
      Zhang, Chiyu  and
      Abdul-Mageed, Muhammad  and
      Aji, Alham Fikri",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.57",
    pages = "944--964"
}
```

## Abstract
Large language models (LLMs) with instruction fine-tuning demonstrate superior generative capabilities. However, these models are resource-intensive. To alleviate this issue, we explore distilling knowledge from instruction-tuned LLMs into much smaller ones. While other similar works have been done, they are often conducted on a limited set of (usually still large) models and are not accompanied by proper evaluations. To this end, we carefully develop a large set of 2.58M instructions based on both existing and newly-generated instructions. In addition to being sizable, we design our instructions to cover a broad set of topics to ensure diversity. Extensive analysis of our instruction dataset confirms its diversity, and we generate responses for these instructions using gpt-3.5-turbo. Leveraging these instructions, we fine-tune a diverse herd of models, collectively referred to as LaMini-LM, which includes models from both the encoder-decoder and decoder-only families, with varying sizes. We evaluate the performance of our models using automatic metrics on 15 different natural language processing (NLP) benchmarks, as well as through human assessment. We also assess the model for hallucination and toxicity, and for the former, we introduce a new benchmark dataset for hallucination-inducing QA. The results demonstrate that our proposed LaMini-LM models are comparable to strong baselines while being much smaller in size.