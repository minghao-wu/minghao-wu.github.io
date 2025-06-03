---
title: "Importance-Aware Data Augmentation for Document-Level Neural Machine Translation"
collection: publications
permalink: /publication/eacl-2024-iada
date: 2024-03-17
venue: 'Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)'
paperurl: 'https://aclanthology.org/2024.eacl-long.44/'
paperurltext: 'Link to ACL anthology'
citation: '<b>Minghao Wu</b>, Yufei Wang, George Foster, Lizhen Qu, and Gholamreza Haffari. 2024. <a href="http://minghao-wu.github.io/files/papers/iada_eacl_2024.pdf"><u>Importance-Aware Data Augmentation for Document-Level Neural Machine Translation</u></a>. In <i>Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)</i>, pages 740–752, St. Julian’s, Malta. Association for Computational Linguistics.'
---

```
@inproceedings{wu-etal-2024-importance,
    title = "Importance-Aware Data Augmentation for Document-Level Neural Machine Translation",
    author = "Wu, Minghao  and
      Wang, Yufei  and
      Foster, George  and
      Qu, Lizhen  and
      Haffari, Gholamreza",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.44",
    pages = "740--752",
    abstract = "Document-level neural machine translation (DocNMT) aims to generate translations that are both coherent and cohesive, in contrast to its sentence-level counterpart. However, due to its longer input length and limited availability of training data, DocNMT often faces the challenge of data sparsity. To overcome this issue, we propose a novel Importance-Aware Data Augmentation (IADA) algorithm for DocNMT that augments the training data based on token importance information estimated by the norm of hidden states and training gradients. We conduct comprehensive experiments on three widely-used DocNMT benchmarks. Our empirical results show that our proposed IADA outperforms strong DocNMT baselines as well as several data augmentation approaches, with statistical significance on both sentence-level and document-level BLEU.",
}
```

## Abstract
Document-level neural machine translation (DocNMT) aims to generate translations that are both coherent and cohesive, in contrast to its sentence-level counterpart. However, due to its longer input length and limited availability of training data, DocNMT often faces the challenge of data sparsity. To overcome this issue, we propose a novel Importance-Aware Data Augmentation (IADA) algorithm for DocNMT that augments the training data based on token importance information estimated by the norm of hidden states and training gradients. We conduct comprehensive experiments on three widely-used DocNMT benchmarks. Our empirical results show that our proposed IADA outperforms strong DocNMT baselines as well as several data augmentation approaches, with statistical significance on both sentence-level and document-level BLEU.