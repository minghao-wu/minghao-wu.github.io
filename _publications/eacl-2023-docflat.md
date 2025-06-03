---
title: "Document Flattening: Beyond Concatenating Context for Document-Level Neural Machine Translation"
collection: publications
permalink: /publication/eacl-2023-docflat
date: 2023-05-02
venue: 'Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics'
paperurl: 'https://aclanthology.org/2023.eacl-main.33'
paperurltext: 'Link to ACL anthology'
citation: '<b>Minghao Wu</b>, George Foster, Lizhen Qu, and Gholamreza Haffari. <a href="https://aclanthology.org/2023.eacl-main.33"><u>Document Flattening: Beyond Concatenating Context for Document-Level Neural Machine Translation</u></a>. <b>EACL 2023<b>.'
---

```
@inproceedings{wu-etal-2023-document,
    title = "Document Flattening: Beyond Concatenating Context for Document-Level Neural Machine Translation",
    author = "Wu, Minghao  and
      Foster, George  and
      Qu, Lizhen  and
      Haffari, Gholamreza",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.33",
    pages = "448--462",
    abstract = "Existing work in document-level neural machine translation commonly concatenates several consecutive sentences as a pseudo-document, and then learns inter-sentential dependencies. This strategy limits the model{'}s ability to leverage information from distant context. We overcome this limitation with a novel Document Flattening (DocFlat) technique that integrates Flat-Batch Attention (FBA) and Neural Context Gate (NCG) into Transformer model to utilizes information beyond the pseudo-document boundaries. FBA allows the model to attend to all the positions in the batch and model the relationships between positions explicitly and NCG identifies the useful information from the distant context. We conduct comprehensive experiments and analyses on three benchmark datasets for English-German translation, and validate the effectiveness of two variants of DocFlat. Empirical results show that our approach outperforms strong baselines with statistical significance on BLEU, COMET and accuracy on the contrastive test set. The analyses highlight that DocFlat is highly effective in capturing the long-range information.",
}
```

## Abstract
Existing work in document-level neural machine translation commonly concatenates several consecutive sentences as a pseudo-document, and then learns inter-sentential dependencies. This strategy limits the model’s ability to leverage information from distant context. We overcome this limitation with a novel Document Flattening (DocFlat) technique that integrates Flat-Batch Attention (FBA) and Neural Context Gate (NCG) into Transformer model to utilize information beyond the pseudo-document boundaries. FBA allows the model to attend to all the positions in the batch and model the relationships between positions explicitly and NCG identifies the useful information from the distant context. We conduct comprehensive experiments and analyses on three benchmark datasets for English-German translation, and validate the effectiveness of two variants of DocFlat. Empirical results show that our approach outperforms strong baselines with statistical significance on BLEU, COMET and accuracy on the contrastive test set. The analyses highlight that DocFlat is highly effective in capturing the long-range information.