---
title: "Bridging the Language Gaps in Large Language Models with Inference-Time Cross-Lingual Intervention"
collection: publications
permalink: /publication/acl-2025-incline
date: 2025-06-01
venue: 'Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)'
paperurl: 'https://arxiv.org/abs/2410.12462'
paperurltext: 'Link to arXiv'
citation: 'Weixuan Wang*, <b>Minghao Wu*</b>, Barry Haddow, and Alexandra Birch. 2024. <a href="https://arxiv.org/abs/2410.12462"><u>Bridging the Language Gaps in Large Language Models with Inference-Time Cross-Lingual Intervention</u></a>. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14226–14240, Miami, Florida, USA. Association for Computational Linguistics.'
---

```
@inproceedings{wu-etal-2024-mixture-skills,
    title = "Bridging the Language Gaps in Large Language Models with Inference-Time Cross-Lingual Intervention",
    author = "Wu, Minghao  and
      Vu, Thuy-Trang  and
      Qu, Lizhen  and
      Haffari, Gholamreza",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.787/",
    doi = "10.18653/v1/2024.emnlp-main.787",
    pages = "14226--14240"
}
```

## Abstract
Large Language Models (LLMs) have shown remarkable capabilities in natural language processing but exhibit significant performance gaps among different languages. Most existing approaches to address these disparities rely on pretraining or fine-tuning, which are resource-intensive. To overcome these limitations without incurring significant costs, we propose Inference-Time Cross-Lingual Intervention (INCLINE), a novel framework that enhances LLM performance on low-performing (source) languages by aligning their internal representations with those of high-performing (target) languages during inference. INCLINE initially learns alignment matrices using parallel sentences from source and target languages through a Least-Squares optimization, and then applies these matrices during inference to transform the low-performing language representations toward the high-performing language space. Extensive experiments on nine benchmarks with five LLMs demonstrate that INCLINE significantly improves performance across diverse tasks and languages, compared to recent strong baselines. Our analysis demonstrates that INCLINE is highly cost-effective and applicable to a wide range of applications.