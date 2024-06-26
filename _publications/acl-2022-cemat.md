---
title: "Universal Conditional Masked Language Pre-training for Neural Machine Translation"
collection: publications
permalink: /publication/acl-2022-cemat
date: 2022-05-22
venue: 'Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)'
paperurl: 'https://aclanthology.org/2022.acl-long.442'
paperurltext: 'Link to ACL anthology'
citation: 'Pengfei Li, Liangyou Li, Meng Zhang, <b>Minghao Wu</b>, and Qun Liu. 2022. <a href="http://minghao-wu.github.io/files/papers/cemat_ACL_2022.pdf"><u>Universal Conditional Masked Language Pre-training for Neural Machine Translation</u></a>. In <i>Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)</i>, pages 6379–6391, Dublin, Ireland. Association for Computational Linguistics.'
---

```
@inproceedings{li-etal-2022-universal,
    title = "Universal Conditional Masked Language Pre-training for Neural Machine Translation",
    author = "Li, Pengfei  and
      Li, Liangyou  and
      Zhang, Meng  and
      Wu, Minghao  and
      Liu, Qun",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.442",
    doi = "10.18653/v1/2022.acl-long.442",
    pages = "6379--6391",
    abstract = "Pre-trained sequence-to-sequence models have significantly improved Neural Machine Translation (NMT). Different from prior works where pre-trained models usually adopt an unidirectional decoder, this paper demonstrates that pre-training a sequence-to-sequence model but with a bidirectional decoder can produce notable performance gains for both Autoregressive and Non-autoregressive NMT. Specifically, we propose CeMAT, a conditional masked language model pre-trained on large-scale bilingual and monolingual corpora in many languages. We also introduce two simple but effective methods to enhance the CeMAT, aligned code-switching {\&} masking and dynamic dual-masking. We conduct extensive experiments and show that our CeMAT can achieve significant performance improvement for all scenarios from low- to extremely high-resource languages, i.e., up to +14.4 BLEU on low resource and +7.9 BLEU improvements on average for Autoregressive NMT. For Non-autoregressive NMT, we demonstrate it can also produce consistent performance gains, i.e., up to +5.3 BLEU. To the best of our knowledge, this is the first work to pre-train a unified model for fine-tuning on both NMT tasks. Code, data, and pre-trained models are available at https://github.com/huawei-noah/Pretrained-Language-Model/CeMAT",
}
```

## Abstract
Pre-trained sequence-to-sequence models have significantly improved Neural Machine Translation (NMT). Different from prior works where pre-trained models usually adopt an unidirectional decoder, this paper demonstrates that pre-training a sequence-to-sequence model but with a bidirectional decoder can produce notable performance gains for both Autoregressive and Non-autoregressive NMT. Specifically, we propose CeMAT, a conditional masked language model pre-trained on large-scale bilingual and monolingual corpora in many languages. We also introduce two simple but effective methods to enhance the CeMAT, aligned code-switching & masking and dynamic dual-masking. We conduct extensive experiments and show that our CeMAT can achieve significant performance improvement for all scenarios from low- to extremely high-resource languages, i.e., up to +14.4 BLEU on low resource and +7.9 BLEU improvements on average for Autoregressive NMT. For Non-autoregressive NMT, we demonstrate it can also produce consistent performance gains, i.e., up to +5.3 BLEU. To the best of our knowledge, this is the first work to pre-train a unified model for fine-tuning on both NMT tasks. Code, data, and pre-trained models are available at https://github.com/huawei-noah/Pretrained-Language-Model/CeMAT