---
title: "Evaluating the Utility of Hand-crafted Features in Sequence Labelling"
collection: publications
permalink: /publication/emnlp-2018-utility
date: 2018-11-02
venue: 'Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP 2018)'
paperurl: 'http://aclweb.org/anthology/D18-1310'
paperurltext: 'Link to ACL anthology'
citation: '<b>Minghao Wu</b>, Fei Liu and Trevor Cohn. <a href="http://aclweb.org/anthology/D18-1310"><u>Evaluating the Utility of Hand-crafted Features in Sequence Labelling</u></a>. <b>EMNLP 2018</b>'
---

```
@inproceedings{wu-etal-2018-evaluating,
    title = "Evaluating the Utility of Hand-crafted Features in Sequence Labelling",
    author = "Wu, Minghao  and
      Liu, Fei  and
      Cohn, Trevor",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D18-1310",
    doi = "10.18653/v1/D18-1310",
    pages = "2850--2856",
    abstract = "Conventional wisdom is that hand-crafted features are redundant for deep learning models, as they already learn adequate representations of text automatically from corpora. In this work, we test this claim by proposing a new method for exploiting handcrafted features as part of a novel hybrid learning approach, incorporating a feature auto-encoder loss component. We evaluate on the task of named entity recognition (NER), where we show that including manual features for part-of-speech, word shapes and gazetteers can improve the performance of a neural CRF model. We obtain a F 1 of 91.89 for the CoNLL-2003 English shared task, which significantly outperforms a collection of highly competitive baseline models. We also present an ablation study showing the importance of auto-encoding, over using features as either inputs or outputs alone, and moreover, show including the autoencoder components reduces training requirements to 60{\%}, while retaining the same predictive accuracy.",
}
```

## Abstract
Conventional wisdom is that hand-crafted features are redundant for deep learning models,
as they already learn adequate representations of text automatically from corpora. In this work, we test this claim by proposing a new method for exploiting handcrafted features as part of a novel hybrid learning approach, incorporating a feature auto-encoder loss component. We evaluate on the task of named entity recognition (NER), where we show that including manual features for part-of-speech, word shapes and gazetteers can improve the performance of a neural CRF model. We obtain a F1 of 91.89 for the CoNLL-2003 English shared task, which significantly outperforms a collection of highly competitive baseline models. We also present an ablation study showing the importance of autoencoding, over using features as either inputs or outputs alone, and moreover, show including the autoencoder components reduces training requirements to 60%, while retaining the same predictive accuracy.