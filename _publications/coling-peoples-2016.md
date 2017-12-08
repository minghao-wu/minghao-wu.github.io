---
title: "Coling Gated End-to-End Memory Networks"
collection: publications
permalink: /publication/eacl-2017-gmemn2n
date: 2017-04-05
venue: 'Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2017)'
paperurl: 'http://www.aclweb.org/anthology/E17-1001'
paperurltext: 'Link to ACL anthology'
citation: '<b>Fei Liu</b> and Julien Perez (2017) <a href="http://liufly.github.io/files/papers/eacl-2017-gmemn2n.pdf"><u>Gated End-to-End Memory Networks</u></a>, In <i>Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2017)</i>, Valencia, Spain, pp. 1-10.'
---

```
@InProceedings{Liu+:2017,
  author    = {Liu, Fei  and  Perez, Julien},
  title     = {Gated End-to-End Memory Networks},
  booktitle = {Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2017)},
  year      = {2017},
  address   = {Valencia, Spain},
  pages     = {1--10}
}
```

## Abstract
Machine reading using differentiable reasoning models has recently shown remarkable progress. In this context, End-to-End trainable Memory Networks (MemN2N) have demonstrated promising performance on simple natural language based reasoning tasks such as factual reasoning and basic deduction. However, other tasks, namely multi-fact question-answering, positional reasoning or dialog related tasks, remain challenging particularly due to the necessity of more complex interactions between the memory and controller modules composing this family of models. In this paper, we introduce a novel end-to-end memory access regulation mechanism inspired by the current progress on the connection short-cutting principle in the field of computer vision. Concretely, we develop a Gated End-to-End trainable Memory Network architecture (GMemN2N). From the machine learning perspective, this new capability is learned in an end-to-end fashion without the use of any additional supervision signal which is, as far as our knowledge goes, the first of its kind. Our experiments show significant improvements on the most challenging tasks in the 20 bAbI dataset, without the use of any domain knowledge. Then, we show improvements on the Dialog bAbI tasks including the real human-bot conversion-based Dialog State Tracking Challenge (DSTC-2) dataset. On these two datasets, our model sets the new state of the art.