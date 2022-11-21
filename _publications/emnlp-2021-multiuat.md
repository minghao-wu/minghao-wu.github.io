---
title: "Uncertainty-Aware Balancing for Multilingual and Multi-Domain Neural Machine Translation Training"
collection: publications
permalink: /publication/emnlp-2021-multiuat
date: 2021-11-07
venue: 'Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP 2021)'
paperurl: 'https://aclanthology.org/2021.emnlp-main.580'
paperurltext: 'Link to ACL anthology'
citation: '<b>Minghao Wu</b>, Yitong Li, Meng Zhang, Liangyou Li, Gholamreza Haffari and Qun Liu (2021) <a href="http://minghao-wu.github.io/files/papers/multiuat_EMNLP_2021.pdf"><u>Uncertainty-Aware Balancing for Multilingual and Multi-Domain Neural Machine Translation Training</u></a>. In <i>Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing</i>, pages 7291–7305, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.'
---

```
@inproceedings{wu-etal-2021-uncertainty,
    title = "Uncertainty-Aware Balancing for Multilingual and Multi-Domain Neural Machine Translation Training",
    author = "Wu, Minghao  and
      Li, Yitong  and
      Zhang, Meng  and
      Li, Liangyou  and
      Haffari, Gholamreza  and
      Liu, Qun",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.580",
    doi = "10.18653/v1/2021.emnlp-main.580",
    pages = "7291--7305",
    abstract = "Learning multilingual and multi-domain translation model is challenging as the heterogeneous and imbalanced data make the model converge inconsistently over different corpora in real world. One common practice is to adjust the share of each corpus in the training, so that the learning process is balanced and low-resource cases can benefit from the high resource ones. However, automatic balancing methods usually depend on the intra- and inter-dataset characteristics, which is usually agnostic or requires human priors. In this work, we propose an approach, MultiUAT, that dynamically adjusts the training data usage based on the model{'}s uncertainty on a small set of trusted clean data for multi-corpus machine translation. We experiments with two classes of uncertainty measures on multilingual (16 languages with 4 settings) and multi-domain settings (4 for in-domain and 2 for out-of-domain on English-German translation) and demonstrate our approach MultiUAT substantially outperforms its baselines, including both static and dynamic strategies. We analyze the cross-domain transfer and show the deficiency of static and similarity based methods.",
}
```

## Abstract
Learning multilingual and multi-domain translation model is challenging as the heterogeneous and imbalanced data make the model converge inconsistently over different corpora in real world. One common practice is to adjust the share of each corpus in the training, so that the learning process is balanced and low-resource cases can benefit from the high resource ones. However, automatic balancing methods usually depend on the intra- and inter-dataset characteristics, which is usually agnostic or requires human priors. In this work, we propose an approach, MultiUAT, that dynamically adjusts the training data usage based on the model{'}s uncertainty on a small set of trusted clean data for multi-corpus machine translation. We experiments with two classes of uncertainty measures on multilingual (16 languages with 4 settings) and multi-domain settings (4 for in-domain and 2 for out-of-domain on English-German translation) and demonstrate our approach MultiUAT substantially outperforms its baselines, including both static and dynamic strategies. We analyze the cross-domain transfer and show the deficiency of static and similarity based methods.